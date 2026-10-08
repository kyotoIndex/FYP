"""Finite postprocessing companion for one explicitly started local experiment."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
import time

LABELS={'hetsa':'原版 HetSA','repair_half':'私人 B 半量适配','repair_full':'私人 B 全量适配',
        'slow_a':'低频共享 A','local_only':'完全本地训练','fixed_a_local_b':'固定 A / 本地 B / 共享头'}

def read_rows(path):
    if not path.exists(): return []
    with path.open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))

def atomic(path,text):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(text,encoding='utf-8'); temporary.replace(path)

def summarize(root):
    status=json.loads((root/'status.json').read_text(encoding='utf-8'))
    rows=read_rows(root/'round_metrics.csv')
    diagnostics=read_rows(root/'client_diagnostics.csv')
    methods=status['methods']; rounds=status.get('rounds',30)
    grouped={m:[r for r in rows if r['method']==m] for m in methods}
    complete={m:r for m,r in grouped.items() if len(r)==rounds}
    stage=status['stage']; finished=sum(len(r) for r in grouped.values())
    text=['# HetSA 新方法实验进度与结果','',
          f'状态：**{stage}**。已完成 {finished}/{len(methods)*rounds} 个方法轮次。',
          f'更新：{datetime.now().astimezone().isoformat(timespec="seconds")}；GPU 训练 PID {status["pid"]}。',
          '', f'协议：AG_NEWS / BERT-base，seed {status["seed"]}，20 客户端，10 个参与，{rounds} 轮，6/9/12 层，rank 16。',
          '', '| 方法 | 进度 | 最近五轮本地准确率 | 最近五轮均衡测试准确率 | 累计实测训练+修复，s | 参数载荷，MiB |',
          '|---|---:|---:|---:|---:|---:|']
    numeric={}
    for method in methods:
        r=grouped[method]
        if not r:
            text.append(f'| {LABELS[method]} | 0/{rounds} | — | — | — | — |'); continue
        local=statistics.mean(float(x['local_accuracy']) for x in r[-5:])*100
        balanced=statistics.mean(float(x['balanced_accuracy']) for x in r[-5:])*100
        last=r[-1]; cost=float(last['cumulative_train_seconds'])+float(last['cumulative_repair_seconds'])
        payload=float(last['cumulative_parameter_payload_mib'])
        text.append(f'| {LABELS[method]} | {len(r)}/{rounds} | {local:.2f}% | {balanced:.2f}% | {cost:.2f} | {payload:.2f} |')
        numeric[method]=dict(completed_rounds=len(r),last5_local_percent=local,last5_balanced_percent=balanced,
                             cumulative_train_and_repair_seconds=cost,cumulative_parameter_payload_mib=payload)
    text+=['','未满 30 轮的指标仅是进度值，不是最终端点；完整方法使用第 26–30 轮平均。',
           '这是一个开发种子的机制探索，不给种子间显著性结论。留出集未用于参数选择。','',
           '## 同一状态下的 A 替换诊断','',
           '| 方法 | A 替换后损失变化 | B 适配后损失变化 |', '|---|---:|---:|']
    for method in methods:
        r=grouped[method]
        if r:
            shock=statistics.mean(float(x['mean_replacement_loss_change']) for x in r)
            repair=statistics.mean(float(x['mean_repair_loss_change']) for x in r)
            text.append(f'| {LABELS[method]} | {shock:+.5f} | {repair:+.5f} |')
    text+=['','损失变化为正表示变差，负表示改善；三种诊断状态固定同一个聚合分类头。',
           'B-only 轮仍交换分类头，不能宣称零通信。完全本地训练的私人分类头与共享头不同。','']
    if 'hetsa' in complete:
        text+=['## 完整方法相对原版 HetSA','',
               '| 方法 | 本地准确率差，pp | 均衡测试差，pp | 训练+修复时间变化 | 参数通信变化 |',
               '|---|---:|---:|---:|---:|']
        ref=numeric['hetsa']
        for method in methods:
            if method=='hetsa' or method not in complete: continue
            item=numeric[method]
            delta=item['last5_local_percent']-ref['last5_local_percent']
            bdelta=item['last5_balanced_percent']-ref['last5_balanced_percent']
            cost=100*(item['cumulative_train_and_repair_seconds']/ref['cumulative_train_and_repair_seconds']-1)
            payload=100*(item['cumulative_parameter_payload_mib']/ref['cumulative_parameter_payload_mib']-1)
            text.append(f'| {LABELS[method]} | {delta:+.2f} | {bdelta:+.2f} | {cost:+.1f}% | {payload:+.1f}% |')
    text+=['','## 可追溯文件','',
           '- `round_metrics.csv`：逐轮效果、实测时间、峰值显存及参数载荷。',
           '- `client_diagnostics.csv`：逐客户端替换/适配损失、能力档位与未参与轮数。',
           '- `participation.csv`、`partition.json`、`initialization.json`：配对参与、数据及初始化。',
           '- `config.json`、`provenance.json`：配置、环境和源文件/数据 SHA-256。',
           '- `latest_state.pt`、各方法 `*_final.pt`：本次实验的训练状态。','',
           '成本是本机真实执行时间与 FP32 参数载荷代理，非物理分布式网络测量。串行轮次计时包括诊断评估、CPU/GPU 拷贝，但不含轮末 checkpoint 序列化；首次骨干/初始 A 分发和协议元数据另计。','']
    if stage=='failed': text+=['## 运行错误','', '```', status.get('error',''), '```','']
    atomic(root/'RESULT.md','\n'.join(text))
    atomic(root/'analysis.json',json.dumps(dict(status=stage,completed_method_rounds=finished,methods=numeric),ensure_ascii=False,indent=2)+'\n')
    return stage

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--results',required=True)
    parser.add_argument('--watch',action='store_true'); args=parser.parse_args()
    root=Path(args.results).resolve(); started=time.monotonic()
    while True:
        try:
            stage=summarize(root)
            if stage in {'completed','failed'} or not args.watch: return
        except (FileNotFoundError,json.JSONDecodeError,KeyError,ValueError):
            if not args.watch: raise
        if time.monotonic()-started>7200:
            atomic(root/'postprocessing_timeout.txt','Postprocessing wait exceeded 2 hours; training was not stopped.\n')
            return
        time.sleep(15)

if __name__=='__main__': main()
