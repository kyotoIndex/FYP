#!/usr/bin/env python3
"""One process, one GPU, one method/seed. No SSH or remote writes."""
from __future__ import annotations
import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from pcla_repair.runtime import run_experiment, POLICIES, atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/pcla_repair/decisive.json")
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--policy",choices=POLICIES,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--donor",type=Path)
    parser.add_argument("--toy",action="store_true")
    parser.add_argument("--stop-after",type=int)
    args=parser.parse_args()
    faulthandler.enable()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but CUDA unavailable; refusing silent CPU fallback")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","4")))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    lock=args.out_dir/".run.lock"
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("run directory is locked; verify that its process has exited before removing a stale lock") from exc
    import platform,time
    os.write(fd,json.dumps({"pid":os.getpid(),"host":platform.node(),"created_unix":time.time()}).encode());os.close(fd)
    try:
        if args.toy:
            from pcla_repair.testing import fixture,tiny_config
            model,layout,clients,capacities,evaluation=fixture(args.seed)
            model.to(device);cfg=tiny_config();provenance={"synthetic_fixture":True}
            partition=None
        else:
            from pcla_repair.bridge import prepare
            path=Path(args.config)
            if not path.is_absolute(): path=REPO/path
            plan=json.loads(path.read_text(encoding="utf-8"))
            cfg=plan["training"]
            model,layout,clients,capacities,evaluation,provenance,partition=prepare(REPO,plan,args.seed,device)
        # All policies get the same source fingerprint. A code change invalidates
        # both resume and donor compatibility, rather than silently mixing runs.
        sources={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((REPO/"pcla_repair").glob("*.py"))}
        sources[str(Path(__file__).relative_to(REPO))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        provenance.update({"code_files":sources,"torch":str(torch.__version__),"cuda":torch.version.cuda})
        rows=run_experiment(model,layout,clients,capacities,evaluation,cfg,args.seed,args.policy,
                            args.out_dir,provenance,donor_path=args.donor,stop_after=args.stop_after,
                            partition=partition)
        print(json.dumps({"synthetic":args.toy,"rounds_saved":len(rows),"output":str(args.out_dir)},ensure_ascii=False))
    except BaseException:
        (args.out_dir/"exception.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise
    finally:
        lock.unlink(missing_ok=True)

if __name__=="__main__": main()
