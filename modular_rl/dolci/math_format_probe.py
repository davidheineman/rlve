"""Inference-only prompt/decoding ablation on the frozen 64 x 8 MATH panel."""

import argparse
import hashlib
import json
import multiprocessing
import os
import socket
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def initialize_grader():
    from modular_rl.v2.math_reward import initialize_worker
    initialize_worker()


def grade_item(item):
    from modular_rl.v2.math_reward import grade
    return grade(*item)


def reason(output):
    value = output['meta_info']['finish_reason']
    return value['type'] if isinstance(value, dict) else str(value)


def generated_ids(output):
    return [int(v[1]) for v in output['meta_info']['output_token_logprobs']]


def run(args):
    import sglang as sgl
    from transformers import AutoTokenizer
    from modular_rl.v2.rollout import forcing_prefix

    args.root.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in args.panel.open() if l.strip()]
    if len(rows) != 64: raise ValueError('Use the frozen 64-prompt panel')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    with socket.socket() as sock:
        sock.bind(('', 0)); port = sock.getsockname()[1]
    engine = sgl.Engine(model_path=str(args.model), tokenizer_path=str(args.model), dtype='bfloat16',
                        tp_size=1, port=port, mem_fraction_static=.3, context_length=4096,
                        max_running_requests=16, cuda_graph_max_bs=16, random_seed=1234,
                        log_level='warning', trust_remote_code=True)
    conditions = [('open-t1', True, 1., 1.), ('closed-t1', False, 1., 1.),
                  ('open-t07', True, .7, .95), ('closed-t07', False, .7, .95)]
    if args.conditions:
        conditions = [value for value in conditions if value[0] in args.conditions]
    report = {'model': str(args.model), 'panel_sha256': hashlib.sha256(args.panel.read_bytes()).hexdigest(),
              'prompts': 64, 'samples_per_prompt': 8, 'sampled_response_cap': 2048,
              'answer_reserve': 256, 'stop_token_ids': [151643, 151645],
              'optimizer_updates': 0, 'conditions': {}}
    try:
        with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize_grader) as pool:
            for name, thinking, temperature, top_p in conditions:
                tick = time.monotonic()
                prompts = [tokenizer.apply_chat_template(row['metadata']['messages'], tokenize=True,
                                add_generation_prompt=True, enable_thinking=thinking) for row in rows for _ in range(8)]
                params = dict(max_new_tokens=1792, temperature=temperature, top_p=top_p,
                              top_k=-1, stop_token_ids=report['stop_token_ids'],
                              skip_special_tokens=False, spaces_between_special_tokens=False)
                outputs = engine.generate(input_ids=prompts, sampling_params=params, return_logprob=True)
                if len(outputs) != 512: raise ValueError('Incomplete evaluation panel')
                responses = [v['text'] for v in outputs]
                counts = [int(v['meta_info']['completion_tokens']) for v in outputs]
                forced = [i for i, v in enumerate(outputs) if reason(v) == 'length']
                if forced:
                    prefixes = [forcing_prefix('math', responses[i], base_model=True) for i in forced]
                    inputs = [prompts[i] + generated_ids(outputs[i]) + tokenizer(prefix, add_special_tokens=False)['input_ids']
                              for i, prefix in zip(forced, prefixes)]
                    second = engine.generate(input_ids=inputs, sampling_params={**params, 'max_new_tokens': 256}, return_logprob=True)
                    for i, prefix, value in zip(forced, prefixes, second):
                        responses[i] += prefix + value['text']
                        counts[i] += int(value['meta_info']['completion_tokens'])
                if any(not 0 < n <= 2048 for n in counts): raise ValueError('Generation cap changed')
                answers = [row['metadata']['target']['answer'] for row in rows for _ in range(8)]
                grades = list(pool.map(grade_item, zip(responses, answers), chunksize=8))
                correct = [bool(v['accuracy']) for v in grades]
                metrics = {'pass@1': sum(correct) / 512, 'pass@8': sum(any(correct[i:i + 8]) for i in range(0, 512, 8)) / 64,
                           'mean_response_tokens': sum(counts) / 512, 'forced_responses': len(forced),
                           'temperature': temperature, 'top_p': top_p, 'enable_thinking': thinking,
                           'seconds': time.monotonic() - tick}
                with (args.root / (name + '.jsonl')).open('w') as stream:
                    for i, (response, grade, n) in enumerate(zip(responses, grades, counts)):
                        stream.write(json.dumps({'key': rows[i // 8]['metadata']['key'], 'sample': i % 8,
                            'response': response, 'sampled_tokens': n, **grade}, ensure_ascii=False) + '\n')
                report['conditions'][name] = metrics
                temporary = args.root / 'report.partial'
                temporary.write_text(json.dumps(report, indent=2) + '\n'); temporary.replace(args.root / 'report.json')
                print('FORMAT_EVAL', name, json.dumps(metrics), flush=True)
    finally:
        engine.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--conditions', nargs='+',
                        choices=['open-t1', 'closed-t1', 'open-t07', 'closed-t07'])
    run(parser.parse_args())
