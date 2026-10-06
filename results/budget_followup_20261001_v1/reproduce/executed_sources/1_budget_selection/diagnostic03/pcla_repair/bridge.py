"""BERT/RoBERTa bridge into the inspected legacy data pipeline.

The old trainer is read-only. IDs are passed as text while splitting, then
resolved to real text; this preserves source-row identity through legacy helpers.
"""
from __future__ import annotations
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys
import torch
from .core import (BudgetLoRALinear, LEGACY_BLOB, keyed_seed, tensor_digest)
from .runtime import fingerprint, seed_train


def checked_legacy(repo: Path):
    source=repo/"src/fedhello_sst2.py"
    raw=source.read_bytes()
    sha=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
    if sha != LEGACY_BLOB:
        raise RuntimeError(f"legacy source changed ({sha}); inspect the diff before adapting the bridge")
    sys.path.insert(0,str(repo/"src"))
    return importlib.import_module("fedhello_sst2")


def prepare(repo: Path, plan: dict, seed: int, device: torch.device):
    legacy=checked_legacy(repo)
    config_bytes=(repo/plan["base_config"]).read_bytes()
    config_blob=hashlib.sha1(b"blob "+str(len(config_bytes)).encode()+b"\0"+config_bytes).hexdigest()
    if config_blob != plan["base_config_blob"]:
        raise RuntimeError("base protocol changed; reconcile its configuration before running")
    old=json.loads(config_bytes.decode("utf-8"))
    old["dataset"].update(plan.get("dataset_overrides",{}))
    if old["dataset"].get("proxy_source") != "train":
        raise ValueError("repaired protocol requires train-sourced proxy")
    def resolve(s):
        p=Path(s); return p if p.is_absolute() else repo/p
    train_path,dev_path=(resolve(old["dataset"][k]) for k in ("train_file","dev_file"))
    train=legacy.read_text_rows(train_path,old["dataset"])
    dev=legacy.read_text_rows(dev_path,old["dataset"])
    registry={f"{split}:{i}":row for split,rs in (("train",train),("dev",dev)) for i,row in enumerate(rs)}
    tags={split:[(f"{split}:{i}",None,legacy.row_label(row)) for i,row in enumerate(rs)]
          for split,rs in (("train",train),("dev",dev))}
    pool,proxy,evaluation=legacy.prepare_data_splits(tags["train"],tags["dev"],old,seed)
    pool,tuning=legacy.split_fixed_proxy_from_train(pool,plan["tuning_samples"],plan["tuning_split_seed"])
    labels=old["dataset"]["num_labels"]
    ratios=[sum(legacy.row_label(row)==j for row in pool)/len(pool) for j in range(labels)]
    specs=legacy.make_clients(old,old["lora"]["num_encoder_layers"],ratios[1],seed,ratios)
    assignments=legacy.build_client_rows(pool,specs,seed)
    ids={str(k):[row[0] for row in rows] for k,rows in assignments.items()}
    allocated=[x for group in ids.values() for x in group]
    proxy_ids=[row[0] for row in proxy]; tuning_ids=[row[0] for row in tuning]
    eval_ids=[row[0] for row in evaluation]
    if len(set(allocated))!=len(allocated) or set(allocated)&(set(proxy_ids)|set(tuning_ids)):
        raise ValueError("source-row reuse or train/proxy/tuning overlap")
    if set(proxy_ids)&set(tuning_ids) or set(eval_ids)&(set(allocated)|set(proxy_ids)|set(tuning_ids)):
        raise ValueError("evaluation split overlaps training/proxy/tuning")
    capacities=[plan["capacity_by_tier"][spec.capability] for spec in specs]
    random.Random(keyed_seed(seed,"resource_assignment")).shuffle(capacities)
    capacities={spec.client_id:cap for spec,cap in zip(specs,capacities)}
    if any(q*plan["training"]["credits_per_layer"] % old["lora"]["rank"] for q in capacities.values()):
        raise ValueError("the primary matched layer-only protocol requires rank-divisible budgets")
    def text_key(row):
        a,b=legacy.row_texts(row)
        return hashlib.sha256(json.dumps([a,b],ensure_ascii=False).encode()).hexdigest()
    train_keys={text_key(registry[x]) for x in allocated}
    eval_keys={text_key(registry[x]) for x in eval_ids}
    partition={"clients":ids,"proxy":proxy_ids,"tuning":tuning_ids,"benchmark_dev":eval_ids,
               "capacities":{str(k):v for k,v in capacities.items()},"train_dev_duplicate_text_count":len(train_keys&eval_keys)}
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    seed_train(keyed_seed(seed,"model_initialization"))
    model_reference=old["dataset"]["model"]
    if (repo/model_reference).is_dir(): model_reference=str(repo/model_reference)
    model=AutoModelForSequenceClassification.from_pretrained(model_reference,
            num_labels=labels,local_files_only=old["dataset"].get("local_files_only",True))
    backbone_hash=tensor_digest(dict(model.state_dict()))
    tokenizer=AutoTokenizer.from_pretrained(model_reference,
            local_files_only=old["dataset"].get("local_files_only",True))
    for layer in legacy.encoder_layers(model):
        for p in ("query","value"):
            base=getattr(layer.attention.self,p)
            setattr(layer.attention.self,p,BudgetLoRALinear(base,old["lora"]["rank"],
                    old["lora"]["alpha"],old["lora"]["dropout"]))
    module_to_layer={id(getattr(layer.attention.self,p)):i
                     for i,layer in enumerate(legacy.encoder_layers(model)) for p in ("query","value")}
    layout={n:module_to_layer[id(m)] for n,m in model.named_modules() if id(m) in module_to_layer}
    def dataset(row_ids):
        return legacy.TextClassificationDataset([registry[x] for x in row_ids],tokenizer,old["dataset"]["max_length"])
    datasets={int(k):dataset(group) for k,group in ids.items()}
    evaluation_ids=tuning_ids if plan["evaluation_split"]=="tuning" else eval_ids
    if plan["evaluation_split"] not in ("tuning","benchmark_dev"):
        raise ValueError("unknown evaluation split")
    provenance={"legacy_blob":LEGACY_BLOB,"base_config":old,
                "train_file_sha256":hashlib.sha256(train_path.read_bytes()).hexdigest(),
                "dev_file_sha256":hashlib.sha256(dev_path.read_bytes()).hexdigest(),
                "partition_sha256":fingerprint(partition),"initial_loaded_model_sha256":backbone_hash,
                "evaluation_split":plan["evaluation_split"],"plan_sha256":fingerprint(plan)}
    return model.to(device),layout,datasets,capacities,dataset(evaluation_ids),provenance,partition
