"""Counterfactual local-update diagnostics on common checkpoints.

Scores, update samples, and evaluation samples are disjoint. All candidates use
one optimizer step with the same learning rate. Results are conditioned on a
checkpoint and are NOT independent federated seeds or additive multi-rank gains.
"""
from __future__ import annotations
import copy
import math
import warnings
import torch
from torch.utils.data import Subset
from .core import (adapter_state,load_adapter,configure,component_scores,
                   lora_modules,SCORE_KINDS,keyed_seed)
from .runtime import batches,to_device,local_train,evaluate


def diagnose(model,layout,dataset,cfg,seed,factor,layer_ids,*,repetition=0):
    batch_size=cfg["probe_batch_size"]
    if len(dataset)<3*batch_size:
        raise ValueError("need three disjoint batches; do not reuse the scoring batch as held-out data")
    if factor not in ("a","b"):
        raise ValueError("choose one trained factor")
    device=next(model.parameters()).device
    rank=next(iter(lora_modules(model).values())).rank
    layers=max(layout.values())+1
    if len(set(layer_ids))!=len(layer_ids) or any(x<0 or x>=layers for x in layer_ids):
        raise ValueError("invalid diagnostic layers")
    indices=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(keyed_seed(seed,"diagnostic",repetition))).tolist()
    score_ids,update_ids,heldout_ids=[indices[i*batch_size:(i+1)*batch_size] for i in range(3)]
    scoring,updating,heldout=[Subset(dataset,ids) for ids in (score_ids,update_ids,heldout_ids)]
    baseline=adapter_state(model)
    full=torch.ones(layers,rank,dtype=torch.bool)
    configure(model,layout,full,factor,head=False)
    model.eval();model.zero_grad(set_to_none=True)
    batch=next(iter(batches(scoring,batch_size,0,False)))
    out=model(**to_device(batch,device))
    if not torch.isfinite(out.loss): raise FloatingPointError("nonfinite diagnostic loss")
    out.loss.backward()
    scores={kind:component_scores(model,layout,factor,kind) for kind in SCORE_KINDS}
    scores["random"]=torch.rand(layers,rank,generator=torch.Generator().manual_seed(keyed_seed(seed,"random-score",repetition)))
    model.zero_grad(set_to_none=True)
    baseline_loss=evaluate(model,layout,heldout,batch_size,device)["loss"]
    local_cfg=copy.deepcopy(cfg)
    local_cfg.update(train_head=False,local_epochs=1,batch_size=batch_size)
    rows=[]
    try:
        for layer in layer_ids:
            for component in range(rank):
                load_adapter(model,baseline)
                support=torch.zeros(layers,rank,dtype=torch.bool);support[layer,component]=True
                local_train(model,layout,updating,support,factor,keyed_seed(seed,"candidate-update",repetition),local_cfg,device)
                loss=evaluate(model,layout,heldout,batch_size,device)["loss"]
                rows.append({"layer":layer,"component":component,"factor":factor,"repetition":repetition,
                             "baseline_loss":baseline_loss,"after_loss":loss,"heldout_gain":baseline_loss-loss,
                             **{kind:float(values[layer,component]) for kind,values in scores.items()}})
    finally:
        load_adapter(model,baseline)
        model.zero_grad(set_to_none=True)
    split={"score_indices":score_ids,"update_indices":update_ids,"heldout_indices":heldout_ids}
    return rows,split


def summarize_within_layer(rows):
    from scipy.stats import spearmanr,kendalltau
    result=[]
    for layer in sorted({x["layer"] for x in rows}):
        group=[x for x in rows if x["layer"]==layer]
        gains=[x["heldout_gain"] for x in group]
        best=max(gains)
        for kind in (*SCORE_KINDS,"random"):
            scores=[x[kind] for x in group]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho=float(spearmanr(scores,gains).statistic)
                tau=float(kendalltau(scores,gains).statistic)
            selected=max(range(len(group)),key=lambda i:(scores[i],-group[i]["component"]))
            constant=max(scores)==min(scores)
            result.append({"layer":layer,"score":kind,"candidates":len(group),
                           "spearman":rho if math.isfinite(rho) else None,
                           "kendall":tau if math.isfinite(tau) else None,
                           "top1_regret":best-gains[selected],"selected_gain":gains[selected],
                           "best_single_component_gain":best,"constant_score":constant})
    return result
