#!/usr/bin/env python3
"""Rebuild all figures from public original-value exports; no training required."""
from pathlib import Path
import csv, gzip, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT = Path(__file__).resolve().parent
DATA = OUT.parent / 'data'
INPUTS = []
T95 = {12: 2.200985160082949, 3: 4.302652729696142}
COLORS = ['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7','#56B4E9','#000000','#777777','#555555']
M1 = ['Original-coordinate score','Original-coordinate cyclic','Gradient-rotated score','Gradient-rotated cyclic','Shared-gradient selection','Joint ten-client selection','Full rank4','Full rank8','Full rank16','Alternating full rank16','Two-factor half-quota score','Whole-layer / energy selection','Fed-SB','FedSVD','Fed-PLoRA','Quota-matched random replay']
M2 = {'U002':'Full rank8','U010':'Fixed head-update shrinkage','U014':'Scalar controller (4 levels)','U018':'Two-block controller (16 pairs)','U020':'Separately selected two-block controller','U023':'Server head fitting','U027':'FedLAW','U004':'Full rank4','U006':'Fed-SB'}
ORDER = list(M2)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.titleweight':'bold','axes.labelsize':10,'legend.fontsize':8,'legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.18,'grid.linewidth':0.5,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
AUDIT = {}
def read(rel):
 p=DATA/rel; INPUTS.append(p)
 with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f: return list(csv.DictReader(f))
def interval(values):
 a=np.asarray(values,dtype=float); assert len(a) in T95 and np.isfinite(a).all()
 mean=a.mean(); sd=a.std(ddof=1); return mean,T95[len(a)]*sd/np.sqrt(len(a))
def save(fig,name):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');fig.savefig(OUT/(name+'.png'),bbox_inches='tight');plt.close(fig)
def endpoint_panel(ax,values,labels,title,color):
 means,ci=np.array([interval(v) for v in values]).T
 y=np.arange(len(values));ax.errorbar(means,y,xerr=ci,fmt='o',color=color,capsize=3,ms=4,lw=1.2)
 ax.set_yticks(y,labels);ax.set_ylim(len(values)-.75,-.75);ax.set_xlabel('Returned-model accuracy (%)');ax.set_title(title,loc='left');ax.set_xlim(78,94)
 return means.tolist()
def timing_panel(ax,values,labels,title,color):
 y=np.arange(len(values));
 for i,v in enumerate(values):
  if not v: ax.text(1,i,'Not measured',fontsize=8,color='#666666');continue
  mean=np.mean(v)/60;err=interval(v)[1]/60 if len(v)>1 else 0
  ax.barh(i,mean,color=color,alpha=.8,height=.55);ax.errorbar(mean,i,xerr=err,fmt='none',ecolor='#333333',capsize=3)
  ax.text(mean+err+0.5,i,f'{mean:.1f}',va='center',fontsize=8)
 ax.set_yticks(y,labels);ax.set_ylim(len(values)-.75,-.75);ax.set_xlim(0,48);ax.set_xlabel('Necessary execution time (min)');ax.set_title(title,loc='left')
def budget():
 a=read('1_budget_selection/validation/v_endpoints.csv');cost=read('1_budget_selection/costs/cost_endpoints.csv');join=read('1_budget_selection/costs/quality_cost_join.csv')
 b=read('2_server_feedback/validation/endpoints.csv');bt=read('2_server_feedback/validation/required_costs.csv')
 vals1=[[float(r['b8_accuracy'])*100 for r in a if int(r['slot'])==s] for s in range(1,17)]
 vals2=[[float(r['raw_B8_accuracy'])*100 for r in b if r['recipe_id']==m] for m in ORDER]
 assert all(len(v)==12 for v in vals1+vals2)
 time1=[]
 for s in range(1,17):
  j=next(r for r in join if int(r['slot'])==s)
  v=[float(r['required_seconds']) for r in cost if r['job_id']==j['COST1_job_id']];assert len(v)==(0 if s==16 else 1);time1.append(v)
 time2=[[float(r['required_prefix_seconds']) for r in bt if r['recipe_id']==m and int(r['seed']) in [601,607,613]] for m in ORDER]
 assert all(len(v)==3 for v in time2)
 fig,axs=plt.subplots(2,2,figsize=(13,12),gridspec_kw={'height_ratios':[16,9],'width_ratios':[1,1]})
 means1=endpoint_panel(axs[0,0],vals1,M1,'a  Budget selection: 12-seed mean ± 95% t interval',COLORS[0]);timing_panel(axs[0,1],time1,['']*16,'b  Separate exclusive RTX 4090D run: seed 401',COLORS[0])
 means2=endpoint_panel(axs[1,0],vals2,list(M2.values()),'c  Server feedback: 12-seed mean ± 95% t interval',COLORS[2]);timing_panel(axs[1,1],time2,['']*9,'d  RTX 4090D: seeds 601, 607, 613; mean ± t95',COLORS[2])
 fig.subplots_adjust(left=.26,right=.98,top=.94,hspace=.3,wspace=.24)
 fig.suptitle('Fixed communication cap: returned-model quality and necessary time',fontsize=14,y=.995)
 fig.text(.26,.015,'Cap = 2,382,653,750 bytes; the last complete model is returned (maximum 200 rounds).\nStudy 1 quality mixes RTX 4090D and A100 runs; timing is separate. Random replay has no cost measurement.',fontsize=9)
 save(fig,'fig_1_fixed_budget');AUDIT['fixed_budget']={'study1_n_per_method':12,'study2_n_per_method':12,'study1_accuracy_means_percent':dict(zip(M1,means1)),'study2_accuracy_means_percent':dict(zip(M2.values(),means2)),'study1_time_n':dict(zip(M1,map(len,time1))),'study2_time_n_per_method':3}
def curves():
 rows=read('2_server_feedback/validation/curves.csv.gz');fig,axs=plt.subplots(2,2,figsize=(11,7.6));highlight=['U002','U023','U014','U018'];styles=['-','--','-','-','--','-','-.',':',':']
 for i,m in enumerate(ORDER):
  rs=[r for r in rows if r['recipe_id']==m and 1<=int(r['round'])<=100]
  seeds=sorted(set(r['seed'] for r in rs));assert len(seeds)==12
  lookup={(r['seed'],int(r['round'])):r for r in rs};assert len(lookup)==1200
  x=np.arange(1,101)
  for col,key in enumerate(['raw_accuracy','raw_NLL']):
   arr=np.array([[float(lookup[(s,int(t))][key]) for t in x] for s in seeds]);arr*=100 if col==0 else 1
   mu=arr.mean(axis=0);ci=T95[12]*arr.std(axis=0,ddof=1)/np.sqrt(12)
   for row in [0,1]:
    if row==1 and m not in highlight:continue
    ax=axs[row,col];ax.plot(x,mu,color=COLORS[i],ls=styles[i],lw=1.4 if row==0 else 1.8,label=M2[m]);ax.fill_between(x,mu-ci,mu+ci,color=COLORS[i],alpha=.07 if row==0 else .13,lw=0)
 for i,ax in enumerate(axs.flat):
  ax.set_xlabel('Communication round');ax.set_ylabel('Accuracy (%)' if i%2==0 else 'Negative log-likelihood (nats / example)');ax.set_title(['a  All nine methods: accuracy','b  All nine methods: NLL','c  Selected methods: late accuracy','d  Selected methods: late NLL'][i],loc='left');ax.set_xlim((1,100) if i<2 else (60,100))
 axs[1,0].set_ylim(78,93);axs[1,1].set_ylim(.18,.75)
 handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='lower center',bbox_to_anchor=(.5,.005),columnspacing=1.6)
 fig.subplots_adjust(left=.08,right=.98,top=.88,bottom=.19,hspace=.42,wspace=.24);fig.suptitle('Server feedback: raw model trajectories',fontsize=14,y=.98);fig.text(.08,.92,'Lines: mean of 12 seeds. Shading: pointwise 95% t interval of the seed mean (df = 11).',fontsize=10)
 save(fig,'fig_2_server_feedback_curves');AUDIT['curves']={'methods':9,'seeds_per_method_round':12,'rounds':[1,100],'rows_used':10800,'interval':'pointwise seed mean t95; df11'}
def switch():
 p=DATA/'3_controller_switch_off/paired_values.jsonl';INPUTS.append(p);rows=[json.loads(l) for l in p.read_text().splitlines()];assert len(rows)==24
 fig,axs=plt.subplots(2,3,figsize=(13.2,8),gridspec_kw={'width_ratios':[1,1,1.1]});rounds=[61,65,70,80];AUDIT['switch_off']={}
 for row,recipe in enumerate(['U014','U018']):
  rs=sorted([r for r in rows if r['parent_recipe']==recipe],key=lambda r:int(r['seed']));assert len(rs)==12
  worst=sorted(rs,key=lambda r:r['rounds']['80']['accuracy_difference_pp'])[:2];bad={r['seed']:COLORS[j+3] for j,r in enumerate(worst)}
  for col,key in enumerate(['accuracy_difference_pp','nll_difference']):
   ax=axs[row,col];vals=np.array([[r['rounds'][str(t)][key] for t in rounds] for r in rs]);mu,ci=np.array([interval(vals[:,j]) for j in range(4)]).T
   for r,v in zip(rs,vals):
    ax.plot(rounds,v,color=bad.get(r['seed'],'#999999'),alpha=1 if r['seed'] in bad else .45,lw=1.5 if r['seed'] in bad else .7,marker='o',ms=2,label=f"Seed {r['seed']}" if r['seed'] in bad else None)
   ax.fill_between(rounds,mu-ci,mu+ci,color=COLORS[0],alpha=.17,lw=0);ax.plot(rounds,mu,'o-',color=COLORS[0],lw=2,ms=4,label='12-seed paired mean ± t95');ax.axhline(0,color='#444444',lw=.8,ls='--');ax.set_xticks(rounds);ax.set_xlabel('Reported checkpoint (round)');ax.set_ylabel('Accuracy difference (pp)' if col==0 else 'NLL difference (nats / example)');ax.set_title(f"{'abcdef'[row*3+col]}  {M2[recipe]}\n{'Accuracy' if col==0 else 'NLL'}: switch off − keep controller",loc='left',fontsize=10)
   if col==0:ax.legend(loc='lower right',fontsize=7)
  ax=axs[row,2];v=np.array([r['rounds']['80']['accuracy_difference_pp'] for r in rs]);mu,ci=interval(v);y=np.arange(12)
  ax.scatter(v,y,c=[bad.get(r['seed'],COLORS[2] if vv>0 else '#777777') for r,vv in zip(rs,v)],s=25);ax.set_yticks(np.arange(13),[str(r['seed']) for r in rs]+['Paired mean']);ax.errorbar(mu,12,xerr=ci,fmt='D',capsize=4,color=COLORS[0],ms=5);ax.invert_yaxis();ax.axvline(0,color='#444444',lw=.8,ls='--');ax.set_xlabel('Round-80 accuracy difference (pp)');ax.set_ylabel('Seed');wins=int((v>0).sum());assert wins==7
  ax.set_title(f"{'cf'[row]}  Round 80: {wins}/12 seeds improve\nMean {mu:+.2f} pp; t95 [{mu-ci:+.2f}, {mu+ci:+.2f}]",loc='left',fontsize=10)
  nll=np.array([r['rounds']['80']['nll_difference'] for r in rs]);nm,nci=interval(nll)
  AUDIT['switch_off'][M2[recipe]]={'n_pairs':12,'round80_positive_accuracy_seeds':wins,'accuracy_mean_pp':mu,'accuracy_t95_pp':[mu-ci,mu+ci],'NLL_mean':nm,'NLL_t95':[nm-nci,nm+nci],'highlighted_seeds':list(bad),'reported_checkpoints':rounds}
 fig.subplots_adjust(left=.07,right=.98,top=.85,bottom=.12,hspace=.5,wspace=.4);fig.suptitle('Switching off the server controller after round 60',fontsize=14,y=.98);fig.text(.07,.925,'24 continuation branches; each controller has 12 matched seeds. Accuracy: higher is better; NLL: lower is better.',fontsize=10);fig.text(.07,.025,'Thin lines: individual paired differences. Blue line / band: paired mean / 95% t interval (df = 11).\nOnly rounds 61, 65, 70, 80 are available in the public paired export; connecting lines guide the eye.',fontsize=9)
 save(fig,'fig_3_controller_switch_off')
if __name__=='__main__':
 budget();curves();switch()
 AUDIT['input_sha256']={str(p.relative_to(OUT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dict.fromkeys(INPUTS)}
 (OUT/'figure_numeric_audit.json').write_text(json.dumps(AUDIT,indent=2)+'\n')
 print(json.dumps(AUDIT['switch_off'],indent=2));print('Saved three vector PDFs and three 300-DPI PNGs.')
