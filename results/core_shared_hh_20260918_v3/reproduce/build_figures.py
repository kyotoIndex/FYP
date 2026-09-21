"""Plot every fixed-horizon result without smoothing or checkpoint selection."""
import argparse,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 rows=list(csv.DictReader((a.analysis/'per_round.csv').open()));ends=list(csv.DictReader((a.analysis/'per_seed_endpoints.csv').open()));assert len(rows)==1200 and len(ends)==40
 policies=['pcla','replay_random_rank','layer_only','random_layer','uniform_cyclic','random','head_only','full_alt']
 colors=['#D55E00','#0072B2','#009E73','#CC79A7','#E69F00','#56B4E9','#333333','#777777']
 styles=['-','--','-.','-','--',':',':','-.'];names=['PCLA','Replay random rank','Layer only','Random layer','Uniform cyclic','Random component','Head only','Full alternate']
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16})
 max_loss=max(float(r['nll']) for r in rows)
 with PdfPages(a.out/'all_seed_curves.pdf') as pdf:
  for seed in [7,13,21,42,87]:
   fig,axs=plt.subplots(2,1,figsize=(10,8),sharex=True,layout='constrained')
   for policy,color,style,name in zip(policies,colors,styles,names):
    rs=sorted([r for r in rows if int(r['seed'])==seed and r['policy']==policy],key=lambda r:int(r['round']));assert len(rs)==30
    x=[int(r['round']) for r in rs]
    for ax,key in zip(axs,['accuracy','nll']):ax.plot(x,[float(r[key]) for r in rs],label=name,color=color,ls=style,lw=1.8)
   for ax in axs:ax.set_xlim(1,30);ax.axvspan(26,30,color='#999999',alpha=.12)
   axs[0].set_ylim(0,1);axs[1].set_ylim(0,max_loss*1.05);axs[0].set_ylabel('Benchmark-dev accuracy');axs[1].set_ylabel('Recorded benchmark-dev NLL');axs[1].set_xlabel('Federated round')
   exposure='used in this LR development' if seed in (7,13) else 'not used in this LR development'
   fig.suptitle(f'Fixed shared-HH core comparison | seed {seed}\n{exposure}; shaded region: prespecified rounds 26–30',fontsize=12)
   handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
   pdf.savefig(fig);fig.savefig(a.out/f'seed_{seed}_curves.png',dpi=220);plt.close(fig)
 fig,ax=plt.subplots(figsize=(11,5.8),layout='constrained')
 for seed,color,marker in zip([7,13,21,42,87],['#0072B2','#56B4E9','#D55E00','#009E73','#CC79A7'],['o','s','^','D','v']):
  y=[100*float(next(r for r in ends if int(r['seed'])==seed and r['policy']==pol)['last5_accuracy']) for pol in policies]
  ax.plot(range(8),y,label=f'Seed {seed}'+(' (LR development)' if seed in (7,13) else ''),color=color,marker=marker,ms=6,lw=.9,ls='--' if seed in (7,13) else '-',mfc='white' if seed in (7,13) else color)
 means=[np.mean([100*float(r['last5_accuracy']) for r in ends if r['policy']==pol]) for pol in policies]
 ax.scatter(range(8),means,c='black',marker='_',s=210,lw=2.3,zorder=8,label='Five-seed arithmetic mean')
 ax.set_xticks(range(8),['PCLA','Replay\nrandom rank','Layer\nonly','Random\nlayer','Uniform\ncyclic','Random\ncomponent','Head\nonly','Full\nalternate']);ax.set_ylim(0,100);ax.set_ylabel('Rounds 26–30 mean benchmark-dev accuracy (%)')
 ax.set_title('Eight fixed policies, all five paired training seeds\nLines connect the same seed; method order is fixed, not a ranking',fontsize=12)
 fig.legend(loc='outside lower center',ncol=3,frameon=False)
 for ext in ('png','pdf'):fig.savefig(a.out/f'paired_endpoints.{ext}',dpi=250)
 plt.close(fig)
 print('Saved five complete seed curve panels, one five-page PDF, and paired endpoint figure.')
if __name__=='__main__':main()
