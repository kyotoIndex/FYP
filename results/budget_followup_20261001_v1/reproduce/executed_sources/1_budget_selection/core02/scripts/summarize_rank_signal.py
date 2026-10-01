#!/usr/bin/env python3
"""Summarize fixed rank diagnostics; seeds, not positions, are replication units."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics as st
import warnings
from scipy.stats import spearmanr, kendalltau

KINDS=['normalized_projection','legacy','gate_shared_qv','gradient_norm','random']


def read(p): return json.loads(p.read_text())
def mean(xs):
    values=[x for x in xs if x is not None]
    return st.mean(values) if values else None
def corr(x,y,kind='spearman'):
    if len(set(x))<2 or len(set(y))<2:return None
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        value=float((spearmanr if kind=='spearman' else kendalltau)(x,y).statistic)
    return value if math.isfinite(value) else None
def top(x):return sorted(range(len(x)),key=lambda j:(-x[j],j))[:4]
def jaccard(x,y):
    a,b=set(top(x)),set(top(y));return len(a&b)/len(a|b)
def writecsv(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|',
                      *('|'+'|'.join(map(str,r))+'|' for r in rows)])+'\n'
def fmt(x):return '未定义' if x is None else f'{x:.4f}'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    protocol=read(a.root/'protocol.json');protocol_sha=hashlib.sha256((a.root/'protocol.json').read_bytes()).hexdigest()
    units=[];metrics=[];stability=[];scaling=[];checkpoints=[];identities=[];index={};null_reasons=[]
    frozen_sources=set();script_hashes=set()
    for seed in protocol['seeds']:
        saved_splits=None;clients=None
        for rnd in protocol['checkpoint_rounds']:
            folder=a.root/f'seed_{seed}/round_{rnd}'
            status=read(folder/'status.json');identity=read(folder/'identity.json')
            assert status['state']=='completed' and status['units']==18 and status['optimizer_steps']==288
            assert status['frozen_parameters_verified'] and status['zero_update_loss_exact'] and status['restored_checkpoint_exact']
            assert identity['seed']==seed and identity['round']==rnd and not identity['benchmark_dev_read']
            assert identity['protocol_sha256']==protocol_sha
            frozen_sources.add(json.dumps(identity['training_sources'],sort_keys=True));script_hashes.add(identity['diagnostic_script_sha256'])
            if saved_splits is None:saved_splits=identity['splits'];clients=identity['clients']
            assert identity['splits']==saved_splits and identity['clients']==clients
            for groups in identity['splits'].values():
                assert sorted(map(len,groups.values()))==[16,16,16,16,64]
                flat=[x for g in groups.values() for x in g]
                assert len(flat)==len(set(flat))==128 and all(x.startswith('train:') for x in flat)
            identities.append(identity)
            paths=list(folder.glob('client_*.json'));assert len(paths)==18
            assert {(x['client'],x['layer_index'],x['factor']) for x in map(read,paths)} == set(itertools.product(clients,[0,5,10],['a','b']))
            for path in sorted(paths):
                raw=read(path);context={k:raw[k] for k in ['seed','round','client','capacity','layer_index','factor']}
                index[seed,rnd,raw['client'],raw['layer_index'],raw['factor']]=raw
                baseline=raw['baseline_losses'];assert len(baseline)==64
                assert [c['rank'] for c in raw['candidates']]==list(range(16))
                gains=[]
                for candidate in raw['candidates']:
                    assert len(candidate['after_losses'])==64
                    gain=st.mean(b-a for b,a in zip(baseline,candidate['after_losses']))
                    assert math.isfinite(gain) and math.isclose(gain,candidate['heldout_gain'],abs_tol=1e-14)
                    gains.append(gain)
                oracle=st.mean(gains[j] for j in top(gains));random_expected=st.mean(gains)
                units.append({**context,'baseline_loss':st.mean(baseline),'min_gain':min(gains),'max_gain':max(gains),
                    'gain_span':max(gains)-min(gains),'gain_sd':st.stdev(gains),'positive_gain_positions':sum(x>0 for x in gains),
                    'oracle_top4_mean_gain':oracle,'random_expected_gain':random_expected,
                    'oracle_advantage_over_random':oracle-random_expected})
                for batch in range(3):
                    assert raw['scaling_checks'][batch]['max_logit_absolute_difference']<=3e-5
                    for kind in KINDS:
                        scores=raw['random_scores'][batch] if kind=='random' else raw['scores'][batch][kind]
                        assert len(scores)==16 and all(math.isfinite(x) for x in scores)
                        selected=top(scores);selected_gain=st.mean(gains[j] for j in selected)
                        rho,tau=corr(scores,gains),corr(scores,gains,'kendall')
                        record={**context,'score_batch':batch,'score':kind,'spearman':rho,'kendall':tau,
                                'top4_overlap_fraction':len(set(selected)&set(top(gains)))/4,
                                'selected_top4_mean_gain':selected_gain,'top4_regret':oracle-selected_gain,
                                'advantage_over_random_expectation':selected_gain-random_expected,
                                'oracle_gap_captured':(selected_gain-random_expected)/(oracle-random_expected) if oracle-random_expected>1e-12 else None,
                                'constant_score':len(set(scores))<2,'constant_gain':len(set(gains))<2,
                                'score_range':max(scores)-min(scores)}
                        metrics.append(record)
                        if rho is None or tau is None:null_reasons.append({**context,'score_batch':batch,'score':kind,'constant_score':record['constant_score'],'constant_gain':record['constant_gain']})
                        if kind!='random':
                            scaled=raw['scaled_scores'][batch][kind]
                            order=lambda values:sorted(range(16),key=lambda j:(-values[j],j))
                            scaling.append({**context,'score_batch':batch,'score':kind,'spearman':corr(scores,scaled),
                                'top4_jaccard':jaccard(scores,scaled),'exact_order_unchanged':order(scores)==order(scaled),
                                'max_score_absolute_difference':max(abs(x-y) for x,y in zip(scores,scaled)),
                                'max_relative_score_difference':max(abs(x-y)/max(abs(x),abs(y),1e-30) for x,y in zip(scores,scaled)),
                                'max_logit_absolute_difference':raw['scaling_checks'][batch]['max_logit_absolute_difference']})
                for kind in KINDS:
                    for i,j in itertools.combinations(range(3),2):
                        x=raw['random_scores'][i] if kind=='random' else raw['scores'][i][kind]
                        y=raw['random_scores'][j] if kind=='random' else raw['scores'][j][kind]
                        stability.append({**context,'score':kind,'batch_a':i,'batch_b':j,'spearman':corr(x,y),'top4_jaccard':jaccard(x,y)})
        for before,after in zip(protocol['checkpoint_rounds'],protocol['checkpoint_rounds'][1:]):
            for client,layer,factor in itertools.product(clients,[0,5,10],['a','b']):
                x=index[seed,before,client,layer,factor];y=index[seed,after,client,layer,factor]
                for kind,batch in itertools.product(KINDS,range(3)):
                    xs=x['random_scores'][batch] if kind=='random' else x['scores'][batch][kind]
                    ys=y['random_scores'][batch] if kind=='random' else y['scores'][batch][kind]
                    checkpoints.append({'seed':seed,'round_before':before,'round_after':after,'client':client,'layer_index':layer,'factor':factor,
                                        'score':kind,'score_batch':batch,'spearman':corr(xs,ys),'top4_jaccard':jaccard(xs,ys)})
    assert len(frozen_sources)==len(script_hashes)==1
    assert len(units)==270 and len(metrics)==4050 and len(stability)==4050 and len(scaling)==3240 and len(checkpoints)==2700
    per_seed=[]
    for seed,kind in itertools.product(protocol['seeds'],KINDS):
        ms=[r for r in metrics if r['seed']==seed and r['score']==kind]
        bs=[r for r in stability if r['seed']==seed and r['score']==kind]
        cs=[r for r in checkpoints if r['seed']==seed and r['score']==kind]
        ss=[r for r in scaling if r['seed']==seed and r['score']==kind]
        per_seed.append({'seed':seed,'score':kind,'units':54,'ranking_records':len(ms),
                        **{k:mean([r[k] for r in ms]) for k in ['spearman','kendall','top4_overlap_fraction','selected_top4_mean_gain','top4_regret','advantage_over_random_expectation','oracle_gap_captured']},
                        'valid_correlations':sum(r['spearman'] is not None for r in ms),
                        'batch_spearman':mean([r['spearman'] for r in bs]),'batch_top4_jaccard':mean([r['top4_jaccard'] for r in bs]),
                        'saved_state_spearman':mean([r['spearman'] for r in cs]),'saved_state_top4_jaccard':mean([r['top4_jaccard'] for r in cs]),
                        'scaling_order_unchanged_fraction':mean([float(r['exact_order_unchanged']) for r in ss]),
                        'scaling_top4_jaccard':mean([r['top4_jaccard'] for r in ss])})
    groups=[];strata=[]
    for group,seeds in [('development',[7,13]),('validation',[21,42,87]),('all_five',[7,13,21,42,87])]:
        for kind in KINDS:
            selected=[r for r in per_seed if r['seed'] in seeds and r['score']==kind]
            keys=[k for k in per_seed[0] if k not in ['seed','score','units','ranking_records','valid_correlations']]
            groups.append({'group':group,'score':kind,'seeds':seeds,**{k:mean([r[k] for r in selected]) for k in keys}})
            for layer,factor in itertools.product([0,5,10],['a','b']):
                selected=[r for r in metrics if r['seed'] in seeds and r['score']==kind and r['layer_index']==layer and r['factor']==factor]
                strata.append({'group':group,'score':kind,'layer_index':layer,'factor':factor,
                    'seeds':seeds,'ranking_records':len(selected),
                    **{k:mean([r[k] for r in selected]) for k in ['spearman','kendall','top4_overlap_fraction','advantage_over_random_expectation','top4_regret']}})
    for name,rows in [('unit_gains',units),('rank_metrics',metrics),('batch_stability',stability),('checkpoint_stability',checkpoints),('scaling',scaling),('per_seed_summary',per_seed),('group_summary',groups),('strata_summary',strata)]:
        writecsv(a.out/f'{name}.csv',rows)
    verification={'pass':True,'jobs':15,'units':270,'optimizer_steps':4320,'ranking_records':4050,
                  'source_and_protocol_identities_match':True,'disjoint_train_only_splits':True,'same_splits_across_saved_states':True,
                  'per_example_gains_recomputed':True,'zero_update_loss_exact_all_jobs':True,'frozen_parameters_verified_all_jobs':True,
                  'undefined_correlations':len(null_reasons),'scaling_max_logit_absolute_difference':max(r['max_logit_absolute_difference'] for r in scaling)}
    (a.out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    (a.out/'undefined_correlations.json').write_text(json.dumps(null_reasons,indent=2)+'\n')
    (a.out/'summary.json').write_text(json.dumps({'groups':groups,'per_seed':per_seed},indent=2,allow_nan=False)+'\n')
    names={'normalized_projection':'当前归一化评分','legacy':'旧乘法评分','gate_shared_qv':'共享 Q/V gate 梯度','gradient_norm':'梯度范数','random':'随机排序'}
    report='# 组件评分与独立样本短期收益\n\n15 份已保存模型状态、270 个比较单元、4,320 次单步更新全部完成。每个单元固定种子、保存轮次、客户端、层和更新因子，逐一测量 16 个组件的更新收益。\n\n'
    for group,title in [('validation','21、42、87：单独报告的验证组'),('development','7、13：候选评分比较组')]:
        report+=f'## {title}\n\n'
        report+=table(['评分','Spearman','Kendall','前四名重合比例','选中位置平均收益','相对随机期望的收益差','前四名选择遗憾'],[
            [names[r['score']],fmt(r['spearman']),fmt(r['kendall']),fmt(r['top4_overlap_fraction']),f"{r['selected_top4_mean_gain']:.8f}",f"{r['advantage_over_random_expectation']:+.8f}",f"{r['top4_regret']:.8f}"] for r in groups if r['group']==group])
    report+='\n收益为独立 64 条客户端样本上的平均交叉熵损失下降，正数表示改善。Spearman/Kendall 衡量评分排序与收益排序的一致性。前四名重合比例的随机期望为 0.25。“相对随机期望的收益差”将评分选出的四个位置与均匀随机选择四个位置的精确期望比较。“选择遗憾”是实际收益最高四个位置的平均收益减去评分选出的四个位置的平均收益；这里平均的是分别更新单个位置的收益。\n\n'
    report+='## 当前归一化评分：逐种子结果\n\n'
    report+=table(['seed','Spearman','Kendall','前四名重合比例','相对随机期望的收益差','跨批次 Jaccard','跨保存状态 Jaccard'],[
        [r['seed'],fmt(r['spearman']),fmt(r['kendall']),fmt(r['top4_overlap_fraction']),f"{r['advantage_over_random_expectation']:+.8f}",fmt(r['batch_top4_jaccard']),fmt(r['saved_state_top4_jaccard'])] for r in per_seed if r['score']=='normalized_projection'])
    report+='\nJaccard 是两组前四名位置的交集数量除以并集数量，1 表示完全相同。跨保存状态比较的是第 5→15、15→30 轮，使用相同客户端和样本。\n\n## 全五种子的排序稳定性与等价缩放\n\n'
    report+=table(['评分','跨批次 Spearman','跨批次 Jaccard','跨保存状态 Spearman','跨保存状态 Jaccard','缩放后完整排序不变比例'],[
        [names[r['score']],fmt(r['batch_spearman']),fmt(r['batch_top4_jaccard']),fmt(r['saved_state_spearman']),fmt(r['saved_state_top4_jaccard']),fmt(r['scaling_order_unchanged_fraction'])] for r in groups if r['group']=='all_five'])
    report+='\n## 同层组件的收益差异\n\n每个单元比较同层 16 个组件；下表先计算该单元的读数，再在种子内对 54 个单元等权平均。“真实前四名”根据实际测得收益选择，用来显示可供评分识别的差异。\n\n'
    report+=table(['seed','单元内最大减最小收益','真实前四名平均收益','随机选择期望收益','两者差值'],[
        [seed,*[f"{st.mean(r[k] for r in units if r['seed']==seed):.8f}" for k in ['gain_span','oracle_top4_mean_gain','random_expected_gain','oracle_advantage_over_random']]] for seed in protocol['seeds']])
    report+=f"\n全部 {len(metrics)} 组评分与收益排序相关性都有定义。等价缩放后的最大模型输出差为 {verification['scaling_max_logit_absolute_difference']:.8g}；未在缩放后的参数上追加训练。\n"
    report+='\n## 当前评分在各层与更新方向的表现\n\n以下列出 21、42、87 三个种子在全部六个层/因子组合上的平均结果，完整三组与五种评分见 [分层汇总](strata_summary.csv)。\n\n'
    report+=table(['层（从 1 起）','更新因子','Spearman','前四名重合比例','相对随机期望的收益差'],[
        [r['layer_index']+1,r['factor'].upper(),fmt(r['spearman']),fmt(r['top4_overlap_fraction']),f"{r['advantage_over_random_expectation']:+.8f}"]
        for r in strata if r['group']=='validation' and r['score']=='normalized_projection'])
    report+='\n[逐位置原始结果](../raw/) · [每个单元的收益范围](unit_gains.csv) · [逐种子汇总](per_seed_summary.csv) · [分层、分因子完整指标](rank_metrics.csv) · [检查结果](verification.json) · [返回诊断说明](../README.md)\n'
    (a.out/'RESULTS.md').write_text(report)
    print(json.dumps(verification))


if __name__=='__main__':main()
