#!/usr/bin/env python3
"""Plot seed-level rank-diagnostic readouts from the published CSV."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with (a.analysis/'per_seed_summary.csv').open() as f:rows=list(csv.DictReader(f))
    methods=['normalized_projection','legacy','gate_shared_qv','gradient_norm','random']
    seeds=[7,13,21,42,87]
    assert len(rows)==25
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    panels=[('spearman','Score vs. gain: Spearman',1,(-.12,1)),
            ('top4_overlap_fraction','Top-4 overlap fraction',1,(0,1)),
            ('advantage_over_random_expectation','Gain above random expectation (loss × 1,000)',1000,None),
            ('batch_top4_jaccard','Across-batch top-4 Jaccard',1,(0,1))]
    for seed,color,marker in zip(seeds,['#0072B2','#56B4E9','#D55E00','#009E73','#CC79A7'],['o','s','^','D','v']):
        for ax,(key,label,mult,ylim) in zip(axes.flat,panels):
            y=[mult*float(next(r for r in rows if int(r['seed'])==seed and r['score']==method)[key]) for method in methods]
            ax.plot(range(5),y,color=color,marker=marker,ms=6,lw=.8,
                    mfc='white' if seed in [7,13] else color,ls='--' if seed in [7,13] else '-',
                    label=f'Seed {seed}'+(' (comparison)' if seed in [7,13] else ' (validation)'))
            ax.set_ylabel(label)
            if ylim:ax.set_ylim(*ylim)
    for ax in axes.flat:
        ax.set_xticks(range(5),['Normalized\n(current)','Legacy','Shared\ngate','Gradient\nnorm','Random'])
        ax.grid(axis='y',alpha=.2)
    axes[0,0].axhline(0,color='#555555',lw=.8,ls=':')
    axes[0,1].axhline(.25,color='#555555',lw=.8,ls=':')
    axes[1,0].axhline(0,color='#555555',lw=.8,ls=':')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Rank-score diagnostics: all five seeds\nEach point averages 54 fixed units and three scoring batches',fontsize=13)
    for ext in ['pdf','png']:fig.savefig(a.out/f'rank_signal_by_seed.{ext}',dpi=240)
    plt.close(fig)


if __name__=='__main__':main()
