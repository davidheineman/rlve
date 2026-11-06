"""Read-only audit of completed MOPD stopping, grading, and update traces."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize(values):
    endings=Counter()
    errors=Counter()
    constraints=defaultdict(list)
    for value in values:
        text=value['response'].rstrip()
        endings['im_end' if text.endswith('<|im_end|>') else
                'endoftext' if text.endswith('<|endoftext|>') else 'none']+=1
        grade=json.loads(value['reward_json'])
        if 'error' in grade:errors[grade['error']]+=1
        for name,passed in zip(grade.get('constraints',[]),grade.get('constraint_passes',[])):
            constraints[name].append(bool(passed))
    return {'responses':len(values),'pass@1':statistics.mean(bool(v['correct']) for v in values),
            'mean_response_tokens':statistics.mean(v['response_tokens'] for v in values),
            'truncated':sum(v['status']=='truncated' for v in values),
            'forced':sum('_modular_v2_forcing' in json.loads(v['metadata_json']) for v in values),
            'endings':dict(endings),'errors':dict(errors),
            'constraints':{k:{'n':len(v),'passed':sum(v)} for k,v in constraints.items()}}


def grade_before_forcing(row):
    from modular_rl.graders import grade_code
    metadata=json.loads(row['metadata_json'])
    forcing=metadata.get('_modular_v2_forcing')
    if not forcing:return {'original':bool(row['correct']),'before_forcing':bool(row['correct']),'forced':False}
    grade=grade_code(row['response'][:forcing['answer_char_start']],metadata['target'])
    return {'key':metadata['key'],'original':bool(row['correct']),
            'before_forcing':bool(grade['accuracy']),'forced':True,'error':grade.get('error')}


def audit(root):
    plan=json.loads((root/'teacher-plan.json').read_text());result={'students':{},'teachers':{}}
    for job in json.loads((root/'students/jobs.json').read_text()):
        bank=job['bank'];policy=Path(job['output'])/'policy';panels=[]
        for path in sorted((policy/'rollouts/eval').glob('*.jsonl')):
            values=rows(path)
            panels.append({'step':int(path.stem),'tasks':{t:summarize([v for v in values if v['source_environment']==t])
                           for t in ('math','code','if','chat')}})
        final=rows(policy/'rollouts/eval/0999.jsonl')
        with ThreadPoolExecutor(4) as pool:
            regrades=list(pool.map(grade_before_forcing,[v for v in final if v['source_environment']=='code']))
        scores=defaultdict(list)
        for entry in rows(policy/'teacher-scores.jsonl'):
            for sample in entry['samples']:
                scores[(entry['rollout_id']//100,sample['task'])].append(sample['mean_teacher_minus_student_logp'])
        probes=defaultdict(list)
        for entry in rows(policy/'update-probes.jsonl'):
            for sample in entry['samples']:
                probes[('math','code','if','chat')[sample['sample']%4]].append(sample['surrogate_improvement'])
        result['students'][bank]={'panels':panels,
            'code_before_forcing':{'original_passes':sum(v['original'] for v in regrades),
                'before_forcing_passes':sum(v['before_forcing'] for v in regrades),
                'rescued':sum(v['before_forcing'] and not v['original'] for v in regrades),
                'lost':sum(v['original'] and not v['before_forcing'] for v in regrades),'details':regrades},
            'teacher_score_windows':[{ 'step_start':window*100,'task':task,'mean_teacher_minus_student_logp':statistics.mean(values)}
                                    for (window,task),values in sorted(scores.items())],
            'objective_probes':{t:{'mean_improvement':statistics.mean(v),'fraction_positive':statistics.mean(x>0 for x in v)}
                                for t,v in probes.items()}}
        for task,name in plan['banks'][bank].items():
            teacher=plan['teachers'][name];paths=sorted((Path(teacher['teacher_run_directory'])/'rollouts/eval').glob('*.jsonl'))
            result['teachers'][name]={'bank':bank,'task':task,'final':summarize(rows(paths[-1]))}
    destination=root/'diagnostics/rollout-audit.json';destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result,indent=2)+'\n')
    for bank,data in result['students'].items():
        print(bank,json.dumps({'final':data['panels'][-1]['tasks'],'regrade':{k:v for k,v in data['code_before_forcing'].items() if k!='details'},
                               'probes':data['objective_probes']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    audit(parser.parse_args().root)
