"""Bounded optimization diagnostics. No allocation or aggregation changes.

Separate rates are experimental controls, not a claimed bug fix. Readouts use
only detached tensors and do not consume the training random-number stream.
"""
from __future__ import annotations
import math
from collections import Counter
from typing import Mapping
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def make_optimizer(model: nn.Module, cfg: dict) -> torch.optim.AdamW:
    """Preserve the original single group when both effective rates are equal."""
    named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not named:
        raise ValueError("no trainable parameters")
    common = float(cfg["learning_rate"])
    head_lr = float(cfg.get("head_learning_rate", common))
    lora_lr = float(cfg.get("lora_learning_rate", common))
    if any(not math.isfinite(x) or x <= 0 for x in (common,head_lr,lora_lr)):
        raise ValueError("learning rates must be finite and positive")
    head, lora = [], []
    for name,param in named:
        if name.startswith("classifier."):
            head.append(param)
        elif name.endswith((".lora_a", ".lora_b")):
            lora.append(param)
        else:
            raise ValueError(f"unexpected trainable parameter: {name}")
    # No change of iteration order, optimizer arithmetic, or defaults in HH.
    if head_lr == lora_lr:
        groups = [p for _,p in named]
        default = head_lr
    else:
        groups = []
        if lora:
            groups.append({"params":lora,"lr":lora_lr})
        if head:
            groups.append({"params":head,"lr":head_lr})
        default = common
    return torch.optim.AdamW(groups,lr=default,betas=tuple(cfg["betas"]),
                             eps=cfg["adam_eps"],weight_decay=cfg["weight_decay"],
                             foreach=False)


class TrainReadout:
    def __init__(self, model: nn.Module):
        self.named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
        self.counts = Counter()
        self.examples, self.steps, self.loss_sum = 0, 0, 0.
        self.head_norm_sum = self.lora_norm_sum = 0.
        self.head_norm_max = self.lora_norm_max = 0.
        self.first_loss = self.last_loss = None

    def observe(self, loss: Tensor, labels: Tensor) -> None:
        n = labels.numel();value = float(loss.detach())
        self.counts.update(labels.detach().cpu().tolist())
        self.examples += n;self.steps += 1;self.loss_sum += value*n
        if self.first_loss is None:
            self.first_loss = value
        self.last_loss = value
        squares = {"head":0.,"lora":0.}
        for name,p in self.named:
            if p.grad is not None:
                group = "head" if name.startswith("classifier.") else "lora"
                squares[group] += float(p.grad.detach().double().square().sum())
        h,l = math.sqrt(squares["head"]),math.sqrt(squares["lora"])
        self.head_norm_sum += h;self.lora_norm_sum += l
        self.head_norm_max = max(self.head_norm_max,h)
        self.lora_norm_max = max(self.lora_norm_max,l)

    def result(self) -> dict:
        return {"optimizer_steps":self.steps,"training_examples_seen":self.examples,
                "training_label_counts":{str(k):v for k,v in sorted(self.counts.items())},
                "online_minibatch_loss_mean":self.loss_sum/max(self.examples,1),
                "first_minibatch_loss":self.first_loss,"last_minibatch_loss":self.last_loss,
                "head_gradient_norm_mean":self.head_norm_sum/max(self.steps,1),
                "lora_gradient_norm_mean":self.lora_norm_sum/max(self.steps,1),
                "head_gradient_norm_max":self.head_norm_max,
                "lora_gradient_norm_max":self.lora_norm_max}


def classification_readout(logits: Tensor, labels: Tensor) -> dict:
    z = logits.detach().cpu().double();y = labels.detach().cpu().long()
    if z.ndim != 2 or y.ndim != 1 or len(z) != len(y) or len(y)==0:
        raise ValueError("invalid evaluation logits/labels")
    if not torch.isfinite(z).all() or (y<0).any() or (y>=z.shape[1]).any():
        raise ValueError("nonfinite logits or invalid labels")
    pred=z.argmax(1);prob=z.softmax(1)
    losses=F.cross_entropy(z,y,reduction="none")
    classes=z.shape[1];confusion=[];recall=[];f1=[]
    for c in range(classes):
        confusion.append([int(((y==c)&(pred==j)).sum()) for j in range(classes)])
        tp=int(((y==c)&(pred==c)).sum());pos=int((y==c).sum());out=int((pred==c).sum())
        recall.append(tp/pos if pos else None)
        f1.append(2*tp/(pos+out) if pos+out else 0.)
    result={"predicted_class_counts":[int((pred==c).sum()) for c in range(classes)],
            "true_class_counts":[int((y==c).sum()) for c in range(classes)],
            "confusion":confusion,"class_recall":recall,"macro_f1":sum(f1)/classes,
            "balanced_accuracy":sum(x for x in recall if x is not None)/sum(x is not None for x in recall),
            "nll_recomputed_float64":float(losses.mean()),
            "class_mean_nll":[float(losses[y==c].mean()) if (y==c).any() else None for c in range(classes)],
            "mean_class_probability":prob.mean(0).tolist(),
            "brier_multiclass":float((prob-F.one_hot(y,classes)).square().sum(1).mean())}
    if classes==2:
        qs=torch.tensor([0.,.1,.5,.9,1.],dtype=torch.float64)
        result["margin_quantiles_0_10_50_90_100"] = torch.quantile(z[:,1]-z[:,0],qs).tolist()
        result["positive_probability_quantiles_0_10_50_90_100"] = torch.quantile(prob[:,1],qs).tolist()
    return result


def adapter_readout(state: Mapping[str,Tensor], before: Mapping[str,Tensor] | None=None) -> dict:
    norms={"head":0.,"a":0.,"b":0.};deltas={k:0. for k in norms}
    for name,value in state.items():
        group="head" if name.startswith("classifier.") else "a" if name.endswith(".lora_a") else "b"
        v=value.detach().cpu().double();norms[group]+=float(v.square().sum())
        if before is not None:
            deltas[group]+=float((v-before[name].double()).square().sum())
    result={"parameter_norms":{k:math.sqrt(v) for k,v in norms.items()}}
    if before is not None:
        result["round_update_norms"]={k:math.sqrt(v) for k,v in deltas.items()}
    if "classifier.bias" in state:
        result["classifier_bias"]=state["classifier.bias"].tolist()
    return result


def dataset_labels(dataset) -> Tensor:
    # Both the real BERT dataset and the tiny offline test fixture are supported.
    for name in ("labels","y"):
        value=getattr(dataset,name,None)
        if torch.is_tensor(value) and len(value)==len(dataset):
            return value.detach().cpu().long()
    return torch.tensor([int(dataset[i]["labels"]) for i in range(len(dataset))])


def constant_references(clients, evaluation) -> dict:
    train=torch.cat([dataset_labels(ds) for ds in clients.values()])
    valid=dataset_labels(evaluation)
    classes=max(int(train.max()),int(valid.max()))+1
    tc=torch.bincount(train,minlength=classes).double();vc=torch.bincount(valid,minlength=classes).double()
    prior=tc/tc.sum()
    if (prior<=0).any():raise ValueError("missing population class")
    majority=int(prior.argmax())
    return {"training_population_counts":tc.long().tolist(),"tuning_counts":vc.long().tolist(),
            "training_prior":prior.tolist(),"constant_training_majority_class":majority,
            "constant_training_majority_accuracy":float(vc[majority]/vc.sum()),
            "constant_training_prior_nll":float(-(vc/vc.sum()*prior.log()).sum())}
