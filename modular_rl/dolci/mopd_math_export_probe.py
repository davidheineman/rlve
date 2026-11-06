"""Read-only 64x8 evaluation of an exported math teacher under MOPD decoding."""

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path


def run(args):
    runtime = args.root/'students/runtime'
    sys.path.insert(0, str(runtime))
    os.environ['MODULAR_RL_RUNTIME'] = str(runtime)
    import sglang as sgl
    from transformers import AutoTokenizer
    from modular_rl.v2.math_reward import grade
    from modular_rl.v2.rollout import forcing_prefix

    plan = json.loads((args.root/'teacher-plan.json').read_text())
    name = plan['banks'][args.bank]['math']
    model = args.root/'teachers/exports'/name/'hf-final'
    selection = [json.loads(s) for s in (args.root/'students/selection/math/val.jsonl').read_text().splitlines() if s.strip()]
    assert len(selection) == 64
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    panel = [row for row in selection for _ in range(8)]
    prompts = [tokenizer.apply_chat_template(row['metadata']['messages'], tokenize=True,
                  add_generation_prompt=True, enable_thinking=False) for row in panel]
    with socket.socket() as sock:
        sock.bind(('', 0)); port = sock.getsockname()[1]
    engine = sgl.Engine(model_path=str(model), tokenizer_path=str(model), dtype='bfloat16',
        tp_size=1, port=port, mem_fraction_static=.35, context_length=8192,
        max_running_requests=16, cuda_graph_max_bs=16, random_seed=1234,
        log_level='warning', trust_remote_code=True)
    started = time.monotonic()
    try:
        params = {'max_new_tokens': 1792, 'temperature': .7, 'top_p': .95, 'top_k': -1,
                  'stop_token_ids': [151643,151645], 'skip_special_tokens': False,
                  'spaces_between_special_tokens': False, 'no_stop_trim': True}
        outputs = engine.generate(input_ids=prompts, sampling_params=params, return_logprob=True)
        responses = [v['text'] for v in outputs]
        counts = [v['meta_info']['completion_tokens'] for v in outputs]
        def finish(value):
            reason = value['meta_info']['finish_reason']
            return reason['type'] if isinstance(reason,dict) else str(reason)
        forced = [i for i,v in enumerate(outputs) if finish(v)=='length']
        if forced:
            prefixes = [forcing_prefix('math', responses[i], base_model=True) for i in forced]
            inputs = [prompts[i]+[int(x[1]) for x in outputs[i]['meta_info']['output_token_logprobs']]
                      +tokenizer.encode(prefix, add_special_tokens=False) for i,prefix in zip(forced,prefixes)]
            continuations = engine.generate(input_ids=inputs, sampling_params={**params,'max_new_tokens':256},return_logprob=True)
            for i,prefix,value in zip(forced,prefixes,continuations):
                responses[i] += prefix+value['text']
                counts[i] += value['meta_info']['completion_tokens'];outputs[i] = value
    finally:
        engine.shutdown()
    destination = args.root/'diagnostics/math-export-probe';destination.mkdir(parents=True,exist_ok=True)
    rows=[]
    for row,response,count,value in zip(panel,responses,counts,outputs):
        rows.append({'key':row['metadata']['key'],'response':response,'response_tokens':count,
                     'finish_reason':finish(value),**grade(response,row['metadata']['target']['answer'])})
    assert len(rows)==512
    (destination/(name+'.jsonl')).write_text(''.join(json.dumps(v)+'\n' for v in rows))
    report={'teacher':name,'bank':args.bank,'task':'math','validation_prompts':64,'samples_per_prompt':8,
            'pass@1':sum(v['accuracy'] for v in rows)/512,
            'pass@8':sum(any(v['accuracy'] for v in rows[i:i+8]) for i in range(0,512,8))/64,
            'mean_tokens':sum(counts)/len(counts),'forced':len(forced),
            'truncated':sum(finish(v)=='length' for v in outputs),
            'temperature':.7,'top_p':.95,'max_sampled_tokens':2048,'enable_thinking':False,
            'stop_token_ids':[151643,151645],'seconds':time.monotonic()-started,'training_updates':0}
    (destination/(name+'-report.json')).write_text(json.dumps(report,indent=2)+'\n')
    print('EXPORTED_MATH_TEACHER',json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--bank',choices=['baseline-sft','task-sft'],required=True)
    run(parser.parse_args())
