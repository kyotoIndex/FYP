"""Descriptive V report; paired t intervals over 12 seeds, no new training."""
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path
from scipy.stats import t
NAMES=['原坐标，本地选位','原坐标，均匀轮换','任务梯度旋转后本地选位','任务梯度旋转后均匀轮换','共享梯度线性选位','联合选择十客户端支持','完整双因子 rank4','完整双因子 rank8','完整双因子 rank16','完整rank16交替更新','本地双因子半位置','整层/能量组选择','Fed-SB','FedSVD','Fed-PLoRA','保持旋转选位配额，随机编号']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def writecsv(p,rows):
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def report(root,out):
 m=json.loads((root/'EXPORT_MANIFEST.json').read_text());assert m['complete'] and m['job_count']==192
 for n,h in m['output_sha256'].items():assert sha(root/n)==h,n
 data=list(csv.DictReader((root/'v_endpoints.csv').open()));assert len(data)==192
 seeds=sorted({int(x['seed']) for x in data});assert len(seeds)==12
 curves={}
 for row in csv.DictReader((root/'v_round_curves.csv').open()):
  if 96<=int(row['round'])<=100:curves.setdefault(row['job_id'],[]).append(float(row['accuracy'])*100)
 values={};slot_rows=[]
 for x in data:
  assert len(curves[x['job_id']])==5
  for point in ['half','b8','r100','r96_100_mean']:
   values[int(x['slot']),int(x['seed']),point]=statistics.mean(curves[x['job_id']]) if point=='r96_100_mean' else 100*float(x[point+'_accuracy'])
 for slot in range(1,17):
  for point in ['half','b8','r100','r96_100_mean']:
   a=[values[slot,s,point] for s in seeds];slot_rows.append({'slot':slot,'method_zh':NAMES[slot-1],'endpoint':point,'n':12,'mean_percent':statistics.mean(a),'sample_sd_pp':statistics.stdev(a)})
 pairs=[(slot,8,'vs_D0_selected_full_rank8') for slot in range(1,17) if slot!=8]+[(3,16,'rank_id_contribution'),(5,1,'shared_linear_vs_local'),(6,5,'joint_vs_linear'),(3,1,'rotation_package_vs_original')]
 records=[];diffs=[];critical=float(t.ppf(.975,11))
 for target,ref,purpose in pairs:
  for point in ['half','b8','r100','r96_100_mean']:
   delta=[values[target,s,point]-values[ref,s,point] for s in seeds];mean=statistics.mean(delta);sd=statistics.stdev(delta);hw=critical*sd/math.sqrt(12)
   records.append({'target_slot':target,'reference_slot':ref,'purpose':purpose,'endpoint':point,'n':12,'mean_delta_pp':mean,'sample_sd_pp':sd,'t95_low_pp':mean-hw,'t95_high_pp':mean+hw,'positive':sum(v>0 for v in delta),'zero':sum(v==0 for v in delta),'negative':sum(v<0 for v in delta)})
   diffs.extend({'target_slot':target,'reference_slot':ref,'endpoint':point,'seed':s,'delta_pp':v} for s,v in zip(seeds,delta))
 assert not out.exists();out.mkdir()
 writecsv(out/'slot_endpoints.csv',slot_rows);writecsv(out/'paired_intervals.csv',records);writecsv(out/'paired_seed_differences.csv',diffs)
 lines=['# 12 seed 验证结果','', '主读数为 B8 通信预算内最后返回模型的准确率。完整双因子 rank8 是 D0 已选定的质量参照；半预算、单独第100轮和第96–100轮均值分别列示。所有值来自 train 内开发评价，不是最终 benchmark 测试。','', '配对差按同 seed 相减；均值和样本标准差以12个 seed 为单位。区间采用配对差的双侧95% Student t 区间（df=11），作为本次描述性分析口径，不声称是原V协议预先唯一指定的算法。','', '|方法|B8准确率|相对完整rank8差（百分点）|95%配对区间|','|---|---:|---:|---|']
 for slot in range(1,17):
  a=next(x for x in slot_rows if x['slot']==slot and x['endpoint']=='b8');row=next((x for x in records if x['target_slot']==slot and x['reference_slot']==8 and x['endpoint']=='b8'),None)
  lines.append(f"|{NAMES[slot-1]}|{a['mean_percent']:.2f}%|"+('参照|—|' if row is None else f"{row['mean_delta_pp']:+.2f}|[{row['t95_low_pp']:+.2f}, {row['t95_high_pp']:+.2f}]|"))
 rank=next(x for x in records if x['purpose']=='rank_id_contribution' and x['endpoint']=='b8')
 lines+=['',f"旋转后的本地选位相对同配额随机编号回放：B8平均差 {rank['mean_delta_pp']:+.3f} 个百分点，12 seed中 {rank['positive']} 正、{rank['zero']} 平、{rank['negative']} 负，配对区间 [{rank['t95_low_pp']:+.3f}, {rank['t95_high_pp']:+.3f}]。回放依赖同seed donor选择数量，是机制对照。",'', '区间和描述统计不代替机制诊断与后续协议裁决。']
 (out/'README.md').write_text('\n'.join(lines)+'\n')
 files={f.name:sha(f) for f in out.iterdir()};manifest={'schema':'pcla-v7-v-descriptive-t-report-1','input_manifest_sha256':sha(root/'EXPORT_MANIFEST.json'),'script_sha256':sha(Path(__file__)),'reference_slot':8,'df':11,'critical_value':critical,'definition':'paired mean +/- t(.975,11)*sample_sd/sqrt(12); no multiplicity adjustment; development description','output_sha256':files};(out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--out',type=Path,required=True);v=a.parse_args();print(json.dumps(report(v.root,v.out),indent=2))
