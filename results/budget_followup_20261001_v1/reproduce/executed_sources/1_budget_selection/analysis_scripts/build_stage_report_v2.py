"""Build a factual phase interpretation and decision questions from audited summaries."""
from pathlib import Path
import csv,json,hashlib,statistics,argparse

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def build(root,out):
 tracked={}
 def readj(rel):
  p=root/rel;tracked[rel]=sha(p);return rows(p)
 def readc(rel):
  p=root/rel;tracked[rel]=sha(p);return list(csv.DictReader(p.open()))
 def check_manifest(rel):
  p=root/rel;m=json.loads(p.read_text());tracked[rel]=sha(p)
  entries=m.get('output_sha256',m.get('outputs',{}))
  for n,v in entries.items():assert sha(p.parent/n)==(v if isinstance(v,str) else v['sha256']),n
 for rel in ['validation_report01/MANIFEST.json','quality_cost_join01/MANIFEST.json','cohort_summary01/SUMMARY_MANIFEST.json','direction_summary01/SUMMARY_MANIFEST.json']:check_manifest(rel)
 for rel in ['operational_evidence01/SOURCE_INDEX.json','operational_evidence01/COLLECTION_RECEIPT.json','PHASE_INDEPENDENT_AUDIT.json']:
  tracked[rel]=sha(root/rel)
 c=readj('cohort_summary01/cohort_seed_descriptive.jsonl');d=readj('direction_summary01/direction_overall_descriptive.jsonl');v=readc('validation_report01/slot_endpoints.csv');pairs=readc('validation_report01/paired_intervals.csv');cost=readc('quality_cost_join01/quality_cost_join.csv')
 def stat(fam,choice,metric):
  selected=[x for x in c if x['family']==fam and x['choice_or_comparison']==choice and x['metric']==metric];assert len(selected)==12 and len({x['seed'] for x in selected})==12
  values=[x['value'] for x in selected];assert all(x is not None for x in values)
  return {'family':fam,'choice':choice,'metric':metric,'n':12,'mean':statistics.mean(values),'sample_sd':statistics.stdev(values),'positive':sum(x>0 for x in values),'zero':sum(x==0 for x in values),'negative':sum(x<0 for x in values),'min':min(values),'max':max(values),'seed_values':{str(x['seed']):x['value'] for x in selected}}
 observations=[]
 def get(f,c,m):
  s=stat(f,c,m);observations.append(s);return s
 norm={k:get('selection16',k,'delta_over_parent_loss') for k in ['cyclic','JL_h32','JQ_xi1','public512_linear','full_gradient_choice','finite_bank_search']}
 ratio=get('correction_A_descriptive','finite_bank_search_minus_local','ratio_of_object_sums')
 ranks={k:get('client_rank',k,'mean_defined_client_value') for k in ['h32_full_four','h32_vs_16','full_vs_16','actual1_vs_16','actual4_vs_16','actual4_minus_actual1']}
 extra={k:get('selection16_paired_contrast',k,'delta') for k in ['JQ_xi1_minus_JL_h32','full_gradient_choice_minus_JQ_xi1','public512_linear_minus_JQ_xi1']}
 b8={int(x['slot']):x for x in v if x['endpoint']=='b8'};q={int(x['slot']):x for x in cost};rank=next(x for x in pairs if x['purpose']=='rank_id_contribution' and x['endpoint']=='b8')
 lines=['# PCLA：有限通信预算阶段结果与下一步判断','', '## 本阶段回答的问题','', '在相同通信上限下，哪些 LoRA 更新方式更有效；共享梯度信息、联合选位和坐标旋转能否把局部更新收益转成全局模型收益。模型为 BERT-base，数据为 SST-2 的训练内开发划分。验证使用12个新 seed、16个条件，共192条训练；另有117项统一成本测量及144个机制诊断对象。','', 'B8 表示2,382,653,750字节的通信上限，返回预算内最后模型；200轮上限先到时按实际字节报告。准确率按各 seed 的预算返回模型计算，再跨12 seed 汇总。第100轮、末5轮均值和半预算单独报告。','', '## 1. 预算内训练效果','', f"完整双因子 rank8 平均准确率为 **{float(b8[8]['mean_percent']):.2f}%**。它在开发阶段已经被选为强质量参照。任务梯度旋转后本地选位为 **{float(b8[3]['mean_percent']):.2f}%**；共享梯度线性选位为 **{float(b8[5]['mean_percent']):.2f}%**，联合选择十客户端支持为 **{float(b8[6]['mean_percent']):.2f}%**。",'', '完整双因子表示每次客户端训练同时更新 LoRA 的两个因子；rank8 是其低秩维数。任务梯度旋转保持有效权重不变，用当前参与客户端的聚合梯度调整潜在坐标。共享线性选位以共享梯度预测各位置收益；联合选择在客户端的本地、轮换和共享线性三份掩码之间选择组合。','', f"在旋转后的方法内部，按分数选择编号相对同层/模块配额的随机编号回放，平均高 **{float(rank['mean_delta_pp']):.3f} 个百分点**，{rank['positive']}/12 seed 为正，描述性95%配对区间为 [{float(rank['t95_low_pp']):.3f}, {float(rank['t95_high_pp']):.3f}]。这支持当前旋转条件下的编号选择贡献；与完整rank8仍有明显差距。回放借用同seed donor的配额，作为机制对照。",'', '完整16条件、逐seed、四种预算/轮次端点和配对区间见 [验证结果与配对表](../validation_report01/)；完整曲线与原始损失、类别计数见 [完整曲线与原始读数](../validation_export02/)。配对区间采用df=11的Student t描述口径；没有将其包装为最终基准确认检验。','', '## 2. 有限候选集合有多少收益空间','', '每个诊断从固定父模型出发，在同一组客户端上实际训练候选更新。有限银行参照先用独立选择集挑组合，再用另一份2048条独立样本测量损失下降。它是已训练候选集合中的参照，不是可免费部署的最优算法。','', '下面是相对本地选位的额外损失下降，先除以同对象父模型损失，再对每seed的8个父状态/锚点/因子对象等权平均，最后汇总12 seed。这里的百分比是父损失比例，不是准确率百分点。','', '|选择方式|额外下降占父损失|正向seed数|','|---|---:|---:|']
 labels={'cyclic':'均匀轮换','JL_h32':'共享梯度线性选位','JQ_xi1':'联合选择','public512_linear':'公开512条方向选位','full_gradient_choice':'完整梯度锁定的候选组合','finite_bank_search':'有限银行参照'}
 for k,s in norm.items():lines.append(f"|{labels[k]}|{s['mean']*100:.2f}%|{s['positive']}/12|")
 lines+=['',f"有限银行用另一种归一化（每seed先求收益差总和/父损失总和）得到 **{ratio['mean']*100:.2f}%**，也为{ratio['positive']}/12正向；两种口径都显示候选集合存在收益空间。简单轮换本身也取得明显收益，不能把全部空间归给复杂共享评分。",'', '## 3. 压缩误差与训练时长错配','', '排序相关系数使用Spearman平均秩。每个客户端先比较固定候选，随后汇总对象和seed；客户端或检查点不作为独立seed。','', '|比较的两个排序|平均相关系数|','|---|---:|']
 ranklabels={'h32_full_four':'h32压缩梯度与完整梯度（四候选）','h32_vs_16':'h32一步预测与16步真实收益（三支持）','full_vs_16':'完整梯度一步预测与16步真实收益（三支持）','actual1_vs_16':'实际1步收益与16步真实收益（三支持）','actual4_vs_16':'实际4步收益与16步真实收益（三支持）'}
 for k,l in ranklabels.items():lines.append(f"|{l}|{ranks[k]['mean']:.3f}|")
 improvement=ranks['actual4_minus_actual1'];public=extra['public512_linear_minus_JQ_xi1'];full=extra['full_gradient_choice_minus_JQ_xi1']
 lines+=['', f"4步相对1步的相关性平均变化为 **{improvement['mean']:+.3f}**，{improvement['positive']}/12 seed提高；它没有展示跨seed一致的明显改善。一步梯度排序与多步实际收益的差距，比h32与完整梯度之间的差距更突出。",'',f"完整梯度锁定组合相对联合选择的实际16步收益差为 {full['mean']:+.6f} loss（{full['positive']}/12正）；公开方向相对联合选择为 {public['mean']:+.6f} loss（{public['positive']}/12正）。这些读数不支持直接把压缩维数或公开方向当成已证实的修复。",'', '## 4. 六种表示与更新尺度','', 'N为原表示，D仅平衡两个因子的尺度，S为QR/SVD坐标，T用任务梯度旋转。两项S对照分别随机混合非零方向、移除可训练零方向。所有比较从同父状态出发，并分别观察原步长和范数匹配后的更新。','', '|本地选位，16步，同父差|原步长收益差（loss）|范数匹配收益差（loss）|','|---|---:|---:|']
 for row in d:
  if row['policy']=='local' and row['prefix']==16:
   lines.append(f"|{row['target']} − {row['reference']}|{row['original']['mean']:+.6f}|{row['normmatched']['mean']:+.6f}|")
 lines+=['', '本批48个“移除零方向”对象全部不活跃；其零差值表示没有实际移除可训练零方向，不能解释为检验后排除了零方向作用。原步长与范数匹配的结果要分开解释；完整梯度T只计算了预测对照，没有追加训练。','', '## 5. 成本和失败记录','',f"独立seed401、独占4090D的必要执行时间：旋转后本地选位 **{float(q[3]['COST1_required_seconds']):.1f} 秒**，完整rank8 **{float(q[8]['COST1_required_seconds']):.1f} 秒**。这是独立成本样本，不能当作12seed配对时间；质量与费用对照见 [质量与成本对照](../quality_cost_join01/)。",'', '117项正式成本队列均最终成功；技术测试、封存恢复、路由领取前失败、候选审计失败和整理脚本失败见[历史操作证据及费用说明](../README_OPERATIONAL_EVIDENCE.md)。原有缓存/RNG/连续占用未采集项保持未知，诊断辅助耗时标记为导出时文件证据，不混入统一成本总数。','', '## 6. 对原协议的判断与给Pro的问题','', '当前旋转/共享评分/联合选择工作点尚未达到相对完整rank8的实用推进标准。有限银行的收益空间成立，但它本身不说明增加摘要维数、换公开方向或加入短试训会有效。','', '1. 修正A：h32与完整梯度平均相关系数高于0.5，完整梯度和公开方向又没有优于联合选择的稳定正向读数。是否同意本阶段不自动启动h128或公开方向替换？若认为有特定条件支持修正，请指定已有分层证据、比较对象和可证伪预测。','2. 修正B：4步相对1步仅5/12 seed相关性提高。原“明显改善”未指定唯一数值门槛；是否同意当前证据不足以启动两步试训开发？如果继续，需说明支持它的具体已有条件和新实验能区分的解释。','3. 若A/B均无直接支持，下一阶段怎样以完整rank8为强基线，针对有限预算训练提出机制不同、可检验的新假设？请先给目标、作用机制、最强简单对照、最小决定性实验与停止条件，再给长程任务及资源安排。','4. 描述统计已并列父损失的两种归一化并保留全部分层结果。请指出哪项口径会改变你的决定，明确后续冻结口径；不要用新的汇总选择掩盖整体对照。','', '上述问题随完整结果与独立审计一起交接。本文件不启动新训练或改变冻结协议。']
 assert not out.exists();out.mkdir();(out/'REPORT.md').write_text('\n'.join(lines)+'\n');(out/'OBSERVATIONS.json').write_text(json.dumps(observations,indent=2)+'\n');manifest={'input_sha256':tracked,'script_sha256':sha(Path(__file__)),'output_sha256':{n:sha(out/n) for n in ['REPORT.md','OBSERVATIONS.json']},'stage_complete':False,'note':'report awaits complete phase audit and operational evidence; no Pro send receipt'};(out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.root,a.out)
