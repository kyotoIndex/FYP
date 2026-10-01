#!/usr/bin/env python3
"""Recompute published tables from saved measurements, without training."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import statistics as st
from scipy.stats import t

NAMES = {'pcla':'PCLA', 'random':'随机组件', 'layer_only':'按重要性选整层',
         'replay_random_rank':'固定层和数量、随机组件编号', 'random_layer':'随机选整层',
         'uniform_cyclic':'均匀轮换组件', 'head_only':'只训练分类头', 'full_alt':'更新全部组件'}
SEEDS = [7,13,21,42,87]


def readcsv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def csvtext(rows):
    out=io.StringIO(newline='')
    writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(rows)
    return out.getvalue()


def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|',
                      *('|'+'|'.join(map(str,row))+'|' for row in rows)])+'\n'


def build(root):
    ends=readcsv(root/'data/per_seed_endpoints.csv')
    rounds=readcsv(root/'data/per_round.csv')
    costs=readcsv(root/'data/budget_and_costs.csv')
    lookup={(int(r['seed']),r['policy']):r for r in ends}
    assert len(lookup)==len(ends)==40 and len(rounds)==1200 and len(costs)==40
    summary=json.loads((root/'data/core_summary.json').read_text())
    for seed in SEEDS:
        for policy in NAMES:
            rs=sorted([r for r in rounds if int(r['seed'])==seed and r['policy']==policy],key=lambda r:int(r['round']))
            assert [int(r['round']) for r in rs]==list(range(1,31))
            raw=json.loads((root/f'runs/seed_{seed}/{policy}/metrics.json').read_text())
            assert [m['round'] for m in raw]==list(range(1,31))
            for x,y in zip(rs,raw):
                assert float(x['accuracy'])==y['accuracy'] and float(x['nll'])==y['loss']
            for key,metric in [('last5_accuracy','accuracy'),('last5_nll','nll'),('last5_balanced_accuracy','balanced_accuracy')]:
                assert math.isclose(float(lookup[seed,policy][key]),st.mean(float(r[metric]) for r in rs[-5:]),abs_tol=1e-12)
            saved=next(r for r in summary['records'] if r['seed']==seed and r['policy']==policy)
            assert math.isclose(saved['last5_accuracy'],float(lookup[seed,policy]['last5_accuracy']),abs_tol=1e-12)
    updates=readcsv(root/'data/client_updates.csv');assert len(updates)==12000
    update_map={(int(r['seed']),r['policy'],int(r['round']),int(r['client'])):r for r in updates}
    assert len(update_map)==12000
    for row in updates:
        bits=f"{int(row['support_hex'],16):0192b}"
        counts=[bits[i:i+16].count('1') for i in range(0,192,16)]
        credits=int(row['credits']);assert sum(counts)==credits
        if row['policy'] in ('layer_only','random_layer'):assert set(counts)<={0,16}
        if row['policy']=='uniform_cyclic':
            q,r=divmod(credits,12);assert set(counts)<={q,q+int(r>0)}
        if row['policy']=='replay_random_rank':
            donor=update_map[int(row['seed']),'pcla',int(row['round']),int(row['client'])]
            other=f"{int(donor['support_hex'],16):0192b}"
            assert counts==[other[i:i+16].count('1') for i in range(0,192,16)]
            assert bits!=other
    for cost in costs:
        selected=[r for r in updates if r['seed']==cost['seed'] and r['policy']==cost['policy']]
        assert len(selected)==300
        for k in ['credits','download_bytes','upload_bytes','probe_seconds','train_seconds']:
            assert math.isclose(sum(float(x[k]) for x in selected),float(cost[k]),rel_tol=1e-12,abs_tol=1e-7)
    endpoints=[];paired=[];budget=[]
    for policy in NAMES:
        values=[100*float(lookup[s,policy]['last5_accuracy']) for s in SEEDS]
        endpoints.append({'policy':policy,**{f'seed_{s}_accuracy_percent':x for s,x in zip(SEEDS,values)},
                          'mean_accuracy_percent':st.mean(values),'sample_sd_pp':st.stdev(values)})
        if policy!='pcla':
            diff=[100*(float(lookup[s,'pcla']['last5_accuracy'])-float(lookup[s,policy]['last5_accuracy'])) for s in SEEDS]
            avg=st.mean(diff);half=float(t.ppf(.975,4))*st.stdev(diff)/math.sqrt(5)
            original=next(r for r in summary['paired_estimates'] if r['comparison']==f'pcla - {policy}')
            assert math.isclose(avg,original['mean_last5_difference_pp'],abs_tol=1e-12)
            for a,b in zip([avg-half,avg+half],original['descriptive_t_interval']):assert math.isclose(a,b,abs_tol=1e-12)
            paired.append({'comparison':f'pcla - {policy}',**{f'seed_{s}_difference_pp':d for s,d in zip(SEEDS,diff)},
                           'mean_difference_pp':avg,'t95_lower_pp':avg-half,'t95_upper_pp':avg+half})
        cs=[r for r in costs if r['policy']==policy]
        budget.append({'policy':policy,**{k:st.mean(float(r[k]) for r in cs) for k in
            ['credits','download_bytes','upload_bytes','serialized_traffic_bytes','probe_seconds','train_seconds']}})
    result='# 完整结果与逐种子比较\n\n每个随机种子先计算第 26–30 轮均值，再对种子等权汇总。accuracy 表示预测正确的比例；表中准确率为百分比，差值和标准差为百分点。\n\n'
    result+=table(['方法','seed 7','seed 13','seed 21','seed 42','seed 87','均值','种子间标准差'],[
        [NAMES[r['policy']],*[f"{r[f'seed_{s}_accuracy_percent']:.4f}%" for s in SEEDS],f"{r['mean_accuracy_percent']:.4f}%",f"{r['sample_sd_pp']:.4f}"] for r in endpoints])
    result+='\n## PCLA 减去各对照\n\n同一种子内先求差。95% t 区间根据五个种子的差值计算，用来展示这次估计的波动范围。\n\n'
    result+=table(['对照','seed 7','seed 13','seed 21','seed 42','seed 87','平均差','95% t 区间'],[
        [NAMES[r['comparison'].split(' - ')[1]],*[f"{r[f'seed_{s}_difference_pp']:+.4f}" for s in SEEDS],f"{r['mean_difference_pp']:+.4f}",f"[{r['t95_lower_pp']:+.4f}, {r['t95_upper_pp']:+.4f}]"] for r in paired])
    result+='\n## 预定分组\n\n7、13 参与过本轮学习率开发；21、42、87 未用于这次学习率选择。两组与全五种子结果同时保留。\n\n'
    result+=table(['方法','7、13 平均准确率','21、42、87 平均准确率'],[
        [NAMES[p],*[f"{100*st.mean(float(lookup[s,p]['last5_accuracy']) for s in group):.4f}%" for group in [[7,13],[21,42,87]]]] for p in NAMES])
    result+='\n[逐种子完整指标](data/per_seed_endpoints.csv)还包含损失（NLL）、正负两类的召回率（各类被正确识别的比例）、平衡准确率（两类召回率的平均）、末五轮最低及最高准确率和最后一轮准确率。全 30 轮均值是辅助读数。\n\n[返回实验说明](README.md) · [主表 CSV](data/endpoint_summary.csv) · [七项差值 CSV](data/paired_comparisons.csv)\n'
    costtext='# 更新预算、通信量与实测时间\n\n下表先累计每次 30 轮训练的费用，再取五个种子的平均。一个组件包含同层 query/value 两个投影中同编号的 rank 分量；每次只更新 A、B 两个因子之一。分类头每次更新 1538 个参数，另外记录。\n\n'
    costtext+=table(['方法','组件更新次数','下载 GiB','上传 GiB','总 GiB','评分秒数','客户端训练秒数'],[
        [NAMES[r['policy']],f"{r['credits']:.1f}",*[f"{r[k]/2**30:.6f}" for k in ['download_bytes','upload_bytes','serialized_traffic_bytes']],f"{r['probe_seconds']:.3f}",f"{r['train_seconds']:.3f}"] for r in budget])
    costtext+='''
GiB = 2³⁰ 字节。通信量根据实际发送内容的序列化大小计算：下载完整的当前 LoRA 和分类头，上传被更新因子的参数增量、分类头和位置记录。预先缓存的基础模型、数据文件、保存模型状态及取回结果的流量另存于执行档案。

评分时间是在客户端训练样本上计算重要性的时间，每次使用 16 条样本。训练时间记录客户端更新参数的计算。实验在 RTX 4090 上逐个模拟客户端，各方法分布在不同节点和执行阶段；此处呈现本次实测费用。

“固定层和数量、随机组件编号”需要先读取同种子 PCLA 的选择记录。表中是该对照自身的费用；生成记录的 PCLA 还需要平均 0.769124 GiB 通信量、259.169 秒训练和 19.588 秒评分。两次运行合计约 1.538248 GiB、524.021 秒训练加评分。

PCLA 累计组件更新数量约为全部组件参照的 30.2%，训练加评分约 278.757 秒，全部组件参照约 235.083 秒。这次实现中，减少更新组件并没有带来相应的时间节约。

[逐种子费用](data/budget_and_costs.csv) · [逐客户端更新记录](data/client_updates.csv) · [全部曲线](figures/all_seed_curves.pdf) · [返回实验说明](README.md)

逐客户端记录中的 `support_hex` 是 12×16 个更新位置按“先层、后组件编号”排列的 192 位掩码，以 48 位十六进制数保存。1 表示更新，0 表示冻结。汇总脚本会重新核对全部 12000 次调用的预算，以及随机编号对照的全部 1500 次逐层数量匹配。
'''
    return {'RESULTS.md':result,'COSTS.md':costtext,'data/endpoint_summary.csv':csvtext(endpoints),
            'data/paired_comparisons.csv':csvtext(paired),'data/mean_costs.csv':csvtext(budget)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('results/core_shared_hh_20260918_v3'))
    p.add_argument('--check',action='store_true')
    a=p.parse_args()
    for name,text in build(a.root).items():
        dest=a.root/name
        if a.check:
            assert dest.read_text()==text, f'Generated table differs: {dest}'
        else:
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
    print('PASS: 40 trajectories, 1200 rounds, 12000 client updates, seven paired comparisons and generated tables')


if __name__=='__main__':main()
