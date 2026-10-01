"""Extend client partitions while keeping every established data role unchanged."""
import copy,json,hashlib
from pathlib import Path
from pcla_research_v7.data import _assign_clients,read_train,digest

def extend_manifest(old,train_file,seeds):
    out=copy.deepcopy(old)
    rows,source_sha=read_train(Path(train_file))
    if source_sha != old['source_sha256']: raise ValueError('source differs')
    by_id={x['id']:x for x in rows}
    for seed in seeds:
        if str(seed) in out['partitions']: continue
        tr,rd,tg,ex=_assign_clients(out['roles']['client_pool'],by_id,int(seed))
        out['partitions'][str(seed)]={'client_train':tr,'client_local_readout':rd,'label_targets':tg}
        out['stratification_exceptions'].extend(ex)
    out['parent_split_sha256']=old['split_sha256']
    out['partition_extension_protocol']='pcla-realized-update-feedback-20260929-v8'
    out['split_sha256']=digest({k:v for k,v in out.items() if k!='split_sha256'})
    assert out['roles']==old['roles']
    assert all(out['partitions'][k]==v for k,v in old['partitions'].items())
    group_sets={k:{out['row_groups'][i] for i in ids} for k,ids in out['roles'].items()}
    keys=list(group_sets)
    if any(group_sets[a]&group_sets[b] for i,a in enumerate(keys) for b in keys[i+1:]): raise ValueError('role group overlap')
    return out

def metadata_view(old_meta,manifest):
    out=copy.deepcopy(old_meta)
    out['parent_identity_sha256']=out['identity_sha256']
    out['identity']['split_sha256']=manifest['split_sha256']
    out['identity_sha256']=digest(out['identity'])
    out['reuse']='same immutable token bytes and role row order; client partitions extended'
    return out
