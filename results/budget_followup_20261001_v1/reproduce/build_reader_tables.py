#!/usr/bin/env python3
"""Build reader-facing tables with scientific method names from archived exports."""
import csv
import gzip
import hashlib
import html
import json
import math
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reader'
P1 = ROOT / 'data/1_budget_selection'
P2 = ROOT / 'data/2_server_feedback'
P3 = ROOT / 'data/3_controller_switch_off'
NAMES1 = {1:'原拆法按分数挑',2:'原拆法轮换',3:'梯度旋转按分数挑',4:'梯度旋转轮换',5:'共享梯度摘要挑选',6:'联合客户端挑选',7:'完整rank4',8:'完整rank8',9:'完整rank16',10:'完整rank16交替训练',11:'每轮两因子各半按分数挑',12:'整层挑选',13:'Fed-SB',14:'FedSVD',15:'Fed-PLoRA',16:'同名额随机编号'}
FAMILY2 = {'mean_r8':'完整rank8','mean_r4':'完整rank4','fedsb_r32':'Fed-SB','fixed_scalar':'固定整体缩小一半','fixed_head':'固定分类头四分之一','lower_head_lr':'客户端分类头低学习率','scalar4':'单系数四档','scalar17':'单系数十七档','block16':'双系数十六组合','separable':'双系数分别选择','headfit':'服务器训练分类头','fedlaw_peft':'FedLAW LoRA适配','fedexp_peft':'FedExP LoRA适配'}
REP = {'N':'原拆法','D':'尺度平衡','S':'SVD重拆','T':'梯度旋转','native_SVD':'FedSVD原生重拆'}
REP.update(S_random_nonzero='SVD后非零子空间随机旋转',S_zero_removed='SVD移除零奇异方向',T_sham='固定随机正交旋转')
METHOD = {'local':'按分数挑','cyclic':'轮换','JL':'共享梯度摘要','JQ':'联合客户端挑选','full_both':'完整同时更新','full_alt':'完整交替更新','tied_local':'Q/V绑定按分数挑','local_both_half':'两因子各半按分数挑','uniform_local':'等名额按分数挑','layer':'整层挑选','energy':'能量挑选','fedsb':'Fed-SB','fedsvd':'FedSVD','fedplora':'Fed-PLoRA'}
METHOD.update(cyclic_both_half='两因子各半轮换',equal_quota_cyclic='等名额轮换')

def rows(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else open(path)) as f:
        return list(csv.DictReader(f))

def write(name, rs):
    assert rs
    with open(OUT/name,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]),lineterminator='\n');w.writeheader();w.writerows(rs)

def paired(a,b):
    assert set(a)==set(b) and len(a)==12
    d=[a[s]-b[s] for s in sorted(a)];m=st.mean(d);h=2.200985160082949*st.stdev(d)/math.sqrt(12)
    return dict(mean=m,t95_low=m-h,t95_high=m+h,positive=sum(x>0 for x in d),negative=sum(x<0 for x in d),zero=sum(x==0 for x in d),n_seeds=12)

def main():
    OUT.mkdir(exist_ok=True)
    settings={r['recipe_id']:r for r in rows(P1/'development/native_and_recipe_settings.csv')}
    dev1=[]
    for r in rows(P1/'development/recipe_117.csv'):
        c=settings[r['recipe_id']]
        dev1.append(dict(method=METHOD[c['method']],representation=REP[c['representation']],rank=c['rank'],client_lr=c['lr'],head_lr=c['head_lr'],refresh_period=c['period'],joint_strength=c['xi'],native_client_ranks=c['native_ranks'],optimizer=c['optimizer'],adapter_weight_decay=c['adapter_weight_decay'],adapter_clip_norm=c['adapter_clip_norm'],factor_schedule=c['factor_schedule'],sketch_dimension=c['sketch_h'],public_training=c['uses_public_training'],seed_count=r['n_seeds'],accuracy_percent=100*float(r['b8_raw_accuracy_mean']),half_accuracy_percent=100*float(r['half_raw_accuracy_mean']),returned_round_mean=r['budget_stop_round_mean'],bytes_mean=r['deployed_bytes_mean']))
    write('component_development.csv',dev1)
    val1=rows(P1/'validation/v_endpoints.csv');acc1={};reader1=[]
    for r in val1:
        slot=int(r['slot']);acc1.setdefault(slot,{})[int(r['seed'])]=100*float(r['b8_accuracy'])
        reader1.append(dict(method=NAMES1[slot],seed=r['seed'],accuracy_percent=100*float(r['b8_accuracy']),NLL=r['b8_loss'],returned_round=r['b8_round'],deployed_bytes=r['b8_deployed_bytes'],half_accuracy_percent=100*float(r['half_accuracy']),round100_accuracy_percent=100*float(r['r100_accuracy']),hardware=r['hardware_family']))
    write('component_validation_seeds.csv',reader1)
    joins=rows(P1/'costs/quality_cost_join.csv');method1=[]
    for r in joins:
        c=settings[r['recipe_id']];slot=int(r['slot'])
        method1.append(dict(method=NAMES1[slot],client_lr=c['lr'],head_lr=c['head_lr'],rank=c['rank'],refresh_period=c['period'],joint_strength=c['xi'],native_client_ranks=c['native_ranks'],accuracy_percent=r['V_mean_B8_accuracy_percent'],sample_sd_pp=r['V_sample_sd_accuracy_pp'],returned_round_mean=r['V_mean_return_round'],deployed_bytes_mean=r['V_mean_deployed_bytes'],independent_seed401_seconds=r['COST1_required_seconds'],cost_cache_unknown_clients=r['COST1_cache_unknown_clients'],cost_rng_status=r['COST1_rng_status']))
    write('component_methods.csv',method1)
    recipes={};dev2=[]
    with gzip.open(P2/'development/accepted_originals.jsonl.gz','rt') as f:
        for line in f:
            d=json.loads(line);rid=d['identity']['recipe_id'];c=d['config_original']['recipe'];recipes[rid]=c
    for r in rows(P2/'development/recipe_summary.csv'):
        c=recipes[r['recipe_id']]
        dev2.append(dict(method=FAMILY2[c['family']],rank=c['rank'],client_lr=c['client_lr'],head_lr=c['head_lr'],server_head_lr=c.get('server_head_lr',''),server_lr=c.get('server_lr',''),server_epsilon=c.get('server_epsilon',''),server_head_steps=c.get('server_head_steps',''),public_training=c['public_training'],seed_count=r['seeds'],accuracy_percent=100*float(r['mean_raw_B8_accuracy']),NLL=r['mean_raw_B8_NLL'],required_seconds=r['mean_original_required_seconds'],selected=r['selected_for_VAL']))
    write('server_development.csv',dev2)
    acc2={};nll2={};val2=rows(P2/'validation/endpoints.csv');reader2=[]
    for r in val2:
        rid=r['recipe_id'];seed=int(r['seed']);acc2.setdefault(rid,{})[seed]=100*float(r['raw_B8_accuracy']);nll2.setdefault(rid,{})[seed]=float(r['raw_B8_NLL'])
        reader2.append(dict(method=FAMILY2[recipes[rid]['family']],seed=seed,accuracy_percent=acc2[rid][seed],NLL=nll2[rid][seed],returned_round=r['returned_round'],necessary_seconds=r['required_prefix_seconds'],hardware=r['hardware_family']))
    write('server_validation_seeds.csv',reader2)
    method2=[]
    for rid in acc2:
        c=recipes[rid];seconds=[float(r['required_prefix_seconds']) for r in val2 if r['recipe_id']==rid and int(r['seed']) in (601,607,613)]
        method2.append(dict(method=FAMILY2[c['family']],rank=c['rank'],client_lr=c['client_lr'],head_lr=c['head_lr'],server_head_lr=c.get('server_head_lr',''),server_lr=c.get('server_lr',''),public_training=c['public_training'],accuracy_percent=st.mean(acc2[rid].values()),sample_sd_pp=st.stdev(acc2[rid].values()),NLL=st.mean(nll2[rid].values()),fixed_three_seed_seconds=st.mean(seconds)))
    write('server_methods.csv',method2)
    pairs=[]
    for a,b in [(3,16),(3,8),(4,3),(7,8)]:pairs.append(dict(study='组件选择',left=NAMES1[a],right=NAMES1[b],metric='accuracy_pp',**paired(acc1[a],acc1[b])))
    for a,b in [('U018','U002'),('U023','U002'),('U010','U002'),('U004','U002'),('U020','U002')]:
        for metric,vals in [('accuracy_pp',acc2),('NLL',nll2)]:pairs.append(dict(study='服务器更新控制',left=FAMILY2[recipes[a]['family']],right=FAMILY2[recipes[b]['family']],metric=metric,**paired(vals[a],vals[b])))
    write('key_paired_comparisons.csv',pairs)
    off=[]
    for line in open(P3/'paired_values.jsonl'):
        d=json.loads(line)
        for rnd,r in d['rounds'].items():
            off.append(dict(method=FAMILY2[recipes[d['parent_recipe']]['family']],seed=d['seed'],round=rnd,on_accuracy_percent=100*r['on']['accuracy'],off_accuracy_percent=100*r['off']['accuracy'],accuracy_difference_pp=r['accuracy_difference_pp'],on_NLL=r['on']['nll'],off_NLL=r['off']['nll'],NLL_difference=r['nll_difference']))
    write('switch_off_seed_pairs.csv',off)
    coordinates=[]
    for line in open(P1/'diagnostics/coordinates/direction_overall_descriptive.jsonl'):
        d=json.loads(line)
        for scale in ('original','normmatched'):
            v=d[scale]
            coordinates.append(dict(method=REP[d['target']],reference=REP[d['reference']],choice=METHOD[d['policy']],local_steps=d['prefix'],scale='原幅度' if scale=='original' else '范数匹配',NLL_benefit_difference=v['mean'],sample_sd=v['sample_sd'],seed_count=v['n_seeds']))
    write('coordinate_diagnostics.csv',coordinates)
    files=sorted(OUT.glob('*.csv'))
    sections=[]
    for f in files:
        rs=rows(f);cols=list(rs[0]);table='<thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in cols)+'</tr></thead><tbody>'
        for r in rs:table+='<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in cols)+'</tr>'
        table+='</tbody>'
        sections.append('<section><h2>'+f.name+'</h2><a href="'+f.name+'">下载 CSV</a><div class="scroll"><table>'+table+'</table></div></section>')
    (OUT/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>联邦 LoRA 研究结果</title><style>body{font:15px system-ui;max-width:1400px;margin:40px auto;padding:0 20px;color:#222}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}thead{background:#eef3f8}.scroll{overflow:auto;max-height:540px}input{padding:12px;width:80%;font:inherit}section{margin:40px 0}a{color:#1767a6}</style><h1>全部配方、验证结果与逐种子差异</h1><p>方法使用科学名称。准确率单位为百分数，差值单位为百分点；NLL为平均交叉熵。配对区间使用12独立种子、自由度11的t95。空单元格保留原记录的缺测。时间口径见研究说明。</p><p><a href="../README.md">研究报告</a> · <a href="../HANDOFF.md">上下文与论文交接</a></p><input id="search" placeholder="输入方法、种子或参数筛选行">'+''.join(sections)+'<script>document.querySelector("#search").addEventListener("input",e=>{const q=e.target.value.toLowerCase();document.querySelectorAll("tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))})</script></html>',encoding='utf-8')
    inputs=[P1/'development/native_and_recipe_settings.csv',P1/'development/recipe_117.csv',P1/'validation/v_endpoints.csv',P1/'costs/quality_cost_join.csv',P1/'diagnostics/coordinates/direction_overall_descriptive.jsonl',P2/'development/accepted_originals.jsonl.gz',P2/'development/recipe_summary.csv',P2/'validation/endpoints.csv',P3/'paired_values.jsonl']
    manifest={'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'outputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files+[OUT/'index.html']}}
    (OUT/'BUILD_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print('Built reader tables:',len(dev1),len(dev2),len(reader1),len(reader2),len(off))

if __name__=='__main__':main()
