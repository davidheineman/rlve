"""Inference-only teacher prompt ablation and EOS likelihood diagnostic."""

import argparse
import json
import os
import socket
import statistics
import time
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def run(args):
    import sglang as sgl
    from transformers import AutoTokenizer
    from modular_rl.graders import grade_code
    from modular_rl.if_grader import grade_if
    from modular_rl.v2.rollout import forcing_prefix

    plan=json.loads((args.root/'teacher-plan.json').read_text())
    name=plan['banks'][args.bank][args.task]
    teacher=plan['teachers'][name]
    model=args.root/'teachers/exports'/name/'hf-final'
    selection=args.root/'students/selection'/args.task/'val.jsonl'
    panel=read_rows(selection)
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    destination=args.root/'diagnostics'/name;destination.mkdir(parents=True,exist_ok=True)
    report={'teacher':name,'bank':args.bank,'task':args.task,'conditions':{},'training_updates':0,
            'selection':str(selection),'responses_per_condition':len(panel),'samples_per_prompt':1,
            'stop_token_ids':[151643,151645],'generation_cap':2048}
    with socket.socket() as sock:
        sock.bind(('',0));port=sock.getsockname()[1]
    engine=sgl.Engine(model_path=str(model),tokenizer_path=str(model),dtype='bfloat16',tp_size=1,
        port=port,mem_fraction_static=.35,context_length=4096,max_running_requests=16,
        cuda_graph_max_bs=16,random_seed=1234,log_level='warning',trust_remote_code=True)
    try:
        for condition,thinking in [('original',True),('mopd-prefill',False)]:
            started=time.monotonic()
            prompts=[tokenizer.apply_chat_template(v['metadata']['messages'],tokenize=True,
                       add_generation_prompt=True,enable_thinking=thinking) for v in panel]
            reserve=512 if args.task=='code' else 0
            params={'max_new_tokens':2048-reserve,'temperature':.7,'top_p':.95,'top_k':-1,
                    'stop_token_ids':[151643,151645],'skip_special_tokens':False,
                    'spaces_between_special_tokens':False,'no_stop_trim':True}
            output=engine.generate(input_ids=prompts,sampling_params=params,return_logprob=True)
            responses=[v['text'] for v in output];graded_responses=list(responses)
            counts=[v['meta_info']['completion_tokens'] for v in output]
            def finish(v):
                reason=v['meta_info']['finish_reason']
                return reason['type'] if isinstance(reason,dict) else str(reason)
            forced=[i for i,v in enumerate(output) if args.task=='code' and finish(v)=='length']
            if forced:
                prefixes=[forcing_prefix('code',responses[i],base_model=True) for i in forced]
                inputs=[prompts[i]+[int(x[1]) for x in output[i]['meta_info']['output_token_logprobs']]
                        +tokenizer.encode(prefix,add_special_tokens=False) for i,prefix in zip(forced,prefixes)]
                second=engine.generate(input_ids=inputs,sampling_params={**params,'max_new_tokens':reserve},return_logprob=True)
                for i,prefix,value in zip(forced,prefixes,second):
                    responses[i]+=prefix+value['text'];graded_responses[i]=prefix+value['text']
                    counts[i]+=value['meta_info']['completion_tokens'];output[i]=value
            values=[]
            for row,prompt,response,graded,n,out in zip(panel,prompts,responses,graded_responses,counts,output):
                grade=(grade_code if args.task=='code' else grade_if)(graded,row['metadata']['target'])
                values.append({'key':row['metadata']['key'],'response':response,'prompt_ids':prompt,
                               'sampled_tokens':n,'finish_reason':finish(out),**grade})
            (destination/(condition+'.jsonl')).write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values))
            report['conditions'][condition]={'pass@1':statistics.mean(v['accuracy'] for v in values),
                'mean_tokens':statistics.mean(counts),'truncated':sum(v['finish_reason']=='length' for v in values),
                'im_end':sum(v['response'].rstrip().endswith('<|im_end|>') for v in values),
                'endoftext':sum(v['response'].rstrip().endswith('<|endoftext|>') for v in values),
                'forced':len(forced),'seconds':time.monotonic()-started}
            (destination/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print('TEACHER_PROBE',name,condition,json.dumps(report['conditions'][condition]),flush=True)
    finally:
        engine.shutdown()
    # Score both stop tokens at the end of known correct original teacher answers.
    import torch
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(model,torch_dtype=torch.bfloat16,
        local_files_only=True,attn_implementation='sdpa').eval().to('cuda')
    original=read_rows(sorted((Path(teacher['teacher_run_directory'])/'rollouts/eval').glob('*.jsonl'))[-1])
    selected=[v for v in original if v['correct']][:16]
    eos=[]
    with torch.inference_mode():
        for row in selected:
            text=row['response'].rstrip()
            marker=next((m for m in ('<|im_end|>','<|endoftext|>') if text.endswith(m)),None)
            if not marker:continue
            text=text[:-len(marker)]
            ids=tokenizer.encode(row['prompt']+text,add_special_tokens=False)
            tensor=torch.tensor([ids],device='cuda')
            logits=model(input_ids=tensor,use_cache=False,logits_to_keep=1).logits[0,-1].float()
            logp=logits.log_softmax(-1)
            eos.append({'key':json.loads(row['metadata_json'])['key'],
                        'endoftext_logp':logp[151643].item(),'im_end_logp':logp[151645].item(),
                        'original_end':marker,'target_text_suffix':text[-100:]})
    report['eos_likelihoods']=eos
    report['eos_summary']={key:statistics.mean(v[key] for v in eos) for key in ('endoftext_logp','im_end_logp')}
    (destination/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('EOS_PROBE',name,json.dumps(report['eos_summary']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--bank',choices=['baseline-sft','task-sft'],required=True)
    parser.add_argument('--task',choices=['code','if'],required=True)
    run(parser.parse_args())
