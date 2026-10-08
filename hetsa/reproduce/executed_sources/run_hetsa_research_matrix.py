"""Six fixed methods x five paired seeds x 30 rounds on one local GPU."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import subprocess
import sys
import time
import traceback
from summarize_hetsa_research import summarize, atomic, LABELS

ROOT=Path(__file__).resolve().parents[1]
METHODS=list(LABELS)
SEEDS=[1201,1301,1401,1501,1601]

def dump(path,value): atomic(path,json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def segments(folder):
    return [folder]+sorted(folder.glob('continuation_*'))

def csv_rows(folder,name):
    rows=[]
    for source in segments(folder):
        path=source/name
        if path.exists():
            with path.open(encoding='utf-8',newline='') as f: rows.extend(csv.DictReader(f))
    return rows

def method_records(folder):
    rows=[]
    for source in segments(folder):
        path=source/'method_summary.json'
        if path.exists(): rows.extend(json.loads(path.read_text(encoding='utf-8')))
    if len({r['method'] for r in rows})!=len(rows): raise RuntimeError('Duplicate completed method records')
    return rows

def report(out,status):
    records=[]; progress=0; total=len(SEEDS)*len(METHODS)*30
    for seed in SEEDS:
        folder=out/f'seed_{seed}'
        for source in segments(folder):
            if (source/'status.json').exists(): summarize(source)
        progress+=len(csv_rows(folder,'round_metrics.csv'))
        for r in method_records(folder):
            if r['seed']!=seed: raise RuntimeError('Summary seed mismatch')
            records.append({**r,'seed':seed})
    text=['# HetSA 新机制全规模实验','',
          f'状态：**{status["stage"]}**；已完成 {progress}/{total} 个方法轮次。',
          '', '协议：6 方法 × 5 配对种子 × 30 轮；每种子 20 个客户端，每轮选 10 个，每客户端 512 样本。',
          '种子：1201 / 1301 / 1401 / 1501 / 1601。参数预先固定，不根据某个种子调参。','',
          '运行方法：原版 HetSA、私人 B 半量适配、私人 B 全量适配、低频共享 A、完全本地、固定 A/本地 B/共享头。','',
          '| 方法 | 完成种子 | 本地端点均值 | 均衡测试均值 | 实测训练+修复，s | 参数载荷，MiB |',
          '|---|---:|---:|---:|---:|---:|']
    by_method={m:{r['seed']:r for r in records if r['method']==m} for m in METHODS}
    for method in METHODS:
        rows=list(by_method[method].values())
        if not rows: text.append(f'| {LABELS[method]} | 0/5 | — | — | — | — |'); continue
        local=st.mean(r['last5_local_accuracy'] for r in rows)*100
        balanced=st.mean(r['last5_balanced_accuracy'] for r in rows)*100
        cost=st.mean(r['train_seconds']+r['repair_seconds'] for r in rows)
        payload=st.mean(r['parameter_payload_mib'] for r in rows)
        text.append(f'| {LABELS[method]} | {len(rows)}/5 | {local:.2f}% | {balanced:.2f}% | {cost:.2f} | {payload:.2f} |')
    comparisons=[]
    if all(len(by_method[m])==5 for m in METHODS):
        text+=['','## 配对结果：相对原版 HetSA','',
               '| 方法 | 本地准确率差，pp | 配对 t 95% CI，pp | 均衡测试差，pp | 训练+修复节省 | 参数载荷节省 |',
               '|---|---:|---|---:|---:|---:|']
        for method in METHODS[1:]:
            diff=[100*(by_method[method][s]['last5_local_accuracy']-by_method['hetsa'][s]['last5_local_accuracy']) for s in SEEDS]
            mean=st.mean(diff); sd=st.stdev(diff); margin=2.7764451051977987*sd/math.sqrt(5)
            balanced=st.mean(100*(by_method[method][s]['last5_balanced_accuracy']-by_method['hetsa'][s]['last5_balanced_accuracy']) for s in SEEDS)
            train_saving=100*(1-st.mean((by_method[method][s]['train_seconds']+by_method[method][s]['repair_seconds'])/(by_method['hetsa'][s]['train_seconds']+by_method['hetsa'][s]['repair_seconds']) for s in SEEDS))
            payload_saving=100*(1-st.mean(by_method[method][s]['parameter_payload_mib']/by_method['hetsa'][s]['parameter_payload_mib'] for s in SEEDS))
            text.append(f'| {LABELS[method]} | {mean:+.2f} | [{mean-margin:+.2f}, {mean+margin:+.2f}] | {balanced:+.2f} | {train_saving:+.1f}% | {payload_saving:+.1f}% |')
            comparisons.append(dict(method=method,seeds=SEEDS,per_seed_difference_pp=diff,mean_difference_pp=mean,ci95_pp=[mean-margin,mean+margin],balanced_difference_pp=balanced,train_and_repair_saving_percent=train_saving,payload_saving_percent=payload_saving))
        text+=['','区间以五个种子为独立单位，未作多重比较校正；不以客户端/轮次充当独立重复。']
    text+=['','## 结果与日志','',
           '- 各 `seed_*` 文件夹：逐轮指标、逐客户端替换诊断、配对分配与状态。',
           '- 各 `seed_*/RESULT.md`：该种子自动生成的明细。',
           '- `matrix_status.json`：当前种子与进度；`seed_*.stdout.log`：训练输出。',
           '- `protocol.json`：预先固定的配置和代码哈希；`matrix_analysis.json`：汇总。','',
           '端点：第 26–30 轮客户端固定留出准确率；均衡测试为共同 256 条测试子集的客户端模型平均。',
           '实测成本与参数载荷分别报告；没有把模拟通信时间当真实速度。固定 A 对照仍共享分类头，完全本地对照的分类头则私有。','']
    if status['stage']=='failed': text+=['## 错误','', '```',status.get('error',''),'```','']
    atomic(out/'RESULT.md','\n'.join(text))
    dump(out/'matrix_analysis.json',dict(stage=status['stage'],completed_method_rounds=progress,records=records,paired_comparisons=comparisons))

def verify(folder):
    rows=csv_rows(folder,'participation.csv')
    paired={m:sorted((int(r['round']),int(r['client_id'])) for r in rows if r['method']==m) for m in METHODS}
    if not all(len(v)==300 and v==paired['hetsa'] for v in paired.values()): raise RuntimeError('Participation pairing failed')
    rows=csv_rows(folder,'round_metrics.csv')
    if len(rows)!=180: raise RuntimeError('Incomplete six-method matrix')
    if len({(r['method'],int(r['round'])) for r in rows})!=180: raise RuntimeError('Duplicate method rounds')
    sources=[s for s in segments(folder) if (s/'partition.json').exists()]
    for name in ['partition.json','initialization.json']:
        if len({hashlib.sha256((s/name).read_bytes()).hexdigest() for s in sources})!=1:
            raise RuntimeError(f'Continuation pairing failed: {name}')
    return dict(participation_pairing=True,methods=6,rounds_each=30,rows=180)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out-dir',required=True)
    parser.add_argument('--resume',action='store_true'); args=parser.parse_args()
    out=(ROOT/args.out_dir).resolve()
    if out.exists() and not args.resume: raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=args.resume); started=time.perf_counter()
    paths=[Path(__file__),ROOT/'scripts/run_hetsa_research.py',ROOT/'src/hetsa_research.py',ROOT/'src/fedhello_sst2.py',ROOT/'configs/hetsa_lora_confirm_extension_v1.json']
    if args.resume:
        old=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
        if old['seeds']!=SEEDS or old['methods']!=METHODS: raise RuntimeError('Protocol mismatch')
        for file in old['files']:
            p=Path(file['path'])
            if p!=Path(__file__) and hashlib.sha256(p.read_bytes()).hexdigest()!=file['sha256']:
                raise RuntimeError(f'Training/input changed: {p}')
        dump(out/f'recovery_{time.time_ns()}.json',dict(reason='Fix duplicate seed keyword in postprocessing; preserve completed training',
            old_status=json.loads((out/'matrix_status.json').read_text(encoding='utf-8')),
            orchestrator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    else:
        dump(out/'protocol.json',dict(date_hkt='2026-10-08',stage='full_scale_mechanism_experiment',seeds=SEEDS,methods=METHODS,
        clients=20,selected_clients=10,samples_per_client=512,rounds=30,
        repair_strengths=[.5,1.],ridge_relative=1e-3,slow_a_warmup=3,slow_a_period=3,
        fixed_hyperparameters=True,endpoint='mean client holdout accuracy rounds 26-30',
            files=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]))
    status=dict(stage='running',seeds=SEEDS,completed_seeds=[],pid=__import__('os').getpid())
    child=None
    try:
        for seed in SEEDS:
            folder=out/f'seed_{seed}'; status.update(current_seed=seed)
            completed={r['method'] for r in method_records(folder)}
            remaining=[m for m in METHODS if m not in completed]
            if not remaining:
                dump(folder/'integrity.json',verify(folder)); status['completed_seeds'].append(seed)
                continue
            target=folder
            if folder.exists():
                prior=csv_rows(folder,'round_metrics.csv')
                if any(r['method'] not in completed for r in prior):
                    raise RuntimeError('Interrupted inside an unfinished method; round-level recovery required')
                target=folder/f'continuation_{len(segments(folder)):02d}'
                original=json.loads((folder/'status.json').read_text(encoding='utf-8'))
                original.update(stage='interrupted',reason='Completed methods preserved; remaining methods continue in separate segment')
                dump(folder/'status.json',original)
            status['active_segment']=str(target)
            dump(out/'matrix_status.json',status)
            suffix='' if target==folder else '_'+target.name
            with (out/f'seed_{seed}{suffix}.stdout.log').open('w',encoding='utf-8') as stdout, (out/f'seed_{seed}{suffix}.stderr.log').open('w',encoding='utf-8') as stderr:
                child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/run_hetsa_research.py'),'--out-dir',str(target),'--seed',str(seed),'--rounds','30','--methods',','.join(remaining)],cwd=ROOT,stdout=stdout,stderr=stderr)
                status['training_pid']=child.pid
                while child.poll() is None:
                    if (target/'status.json').exists():
                        try: status['current']=json.loads((target/'status.json').read_text(encoding='utf-8'))
                        except json.JSONDecodeError: pass
                    status['elapsed_seconds']=time.perf_counter()-started
                    dump(out/'matrix_status.json',status)
                    try: report(out,status)
                    except (FileNotFoundError,KeyError,ValueError,json.JSONDecodeError): pass
                    time.sleep(15)
                if child.returncode: raise RuntimeError(f'seed {seed} exited {child.returncode}; see {target}/status.json')
            dump(folder/'integrity.json',verify(folder))
            status['completed_seeds'].append(seed)
            report(out,status)
        status.update(stage='completed',elapsed_seconds=time.perf_counter()-started)
    except BaseException:
        if child is not None and child.poll() is None:
            child.terminate(); child.wait()
        status.update(stage='failed',error=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started)
        raise
    finally:
        dump(out/'matrix_status.json',status); report(out,status)

if __name__=='__main__': main()
