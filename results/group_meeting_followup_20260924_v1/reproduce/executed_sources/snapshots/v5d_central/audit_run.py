"""Read-only validation and independent metric recomputation. No trust in an 'audited' flag."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import random

import numpy as np
import torch
from common import (ROOT, require, read, sha256, fingerprint, atomic_json, tensor_load,
                    bind_repo, source_files)
from context import validate_partition, labels_from_tsv


def classification(logits, labels):
    z=np.asarray(logits,dtype=np.float64); y=np.asarray(labels,dtype=np.int64)
    require(z.ndim==2 and z.shape[1]==2 and y.shape==(len(z),) and len(y)>0,'invalid binary evaluation')
    require(np.isfinite(z).all() and ((y==0)|(y==1)).all(),'invalid logits/labels')
    pred=z.argmax(1)
    maximum=z.max(1)
    loss=maximum+np.log(np.exp(z-maximum[:,None]).sum(1))-z[np.arange(len(y)),y]
    return prediction_readout(pred.tolist(),y.tolist())|{'nll_float64':float(loss.mean())}


def prediction_readout(predictions, labels):
    pred=np.asarray(predictions,dtype=np.int64); y=np.asarray(labels,dtype=np.int64)
    require(pred.shape==y.shape and len(y)>0 and ((pred==0)|(pred==1)).all(),'invalid predictions')
    confusion=[[int(((y==c)&(pred==k)).sum()) for k in (0,1)] for c in (0,1)]
    recalls=[confusion[c][c]/sum(confusion[c]) if sum(confusion[c]) else None for c in (0,1)]
    return {'accuracy':float((pred==y).mean()),'confusion':confusion,'class_recall':recalls,
            'balanced_accuracy':sum(v for v in recalls if v is not None)/sum(v is not None for v in recalls),
            'predicted_class_counts':[int((pred==k).sum()) for k in (0,1)],
            'one_class_prediction':len(set(pred.tolist()))==1,'samples':len(y),'predictions':pred.tolist()}


def compare_evaluation(ev, labels):
    require(ev['labels']==labels,'evaluation labels/order differs')
    computed=classification(ev['logits'],labels)
    require(ev['predictions']==computed['predictions'],'saved prediction differs from logits')
    require(ev['accuracy']==computed['accuracy'],'accuracy differs from logits')
    require(math.isfinite(ev['loss']),'nonfinite GPU loss')
    require(abs(ev['nll_recomputed_float64']-computed['nll_float64'])<=1e-10,'NLL differs from float64 logits')
    require(ev['confusion']==computed['confusion'] and ev['class_recall']==computed['class_recall'],
            'confusion/recall differs')
    require(abs(ev['balanced_accuracy']-computed['balanced_accuracy'])<1e-14,'balanced accuracy differs')
    return computed|{'gpu_nll':ev['loss']}


def evaluation_identity(out, repo):
    out=Path(out); e=read(out/'evaluation_identity.json');p=read(out/'partition_ids.json')
    validate_partition(p)
    require(fingerprint(p)==e['partition_sha256'],'partition hash differs')
    require(e['source_ids']==p[e['role']],'evaluation source IDs differ')
    if not e['synthetic']:
        require(repo is not None,'real audit needs original TSV paths')
        expected=labels_from_tsv(repo,p,e['data_provenance'],e['role'])
        require(expected==e['labels'],'evaluation label identity does not match TSV')
    else:
        from pcla_repair.testing import TinyDataset
        expected=[int(x['labels']) for x in TinyDataset(200,len(e['labels']))]
        require(expected==e['labels'],'synthetic labels differ from fixture')
    return e,p


def verify_files(out, receipts):
    out=Path(out).resolve()
    for receipt in receipts:
        for rel,digest in receipt.items():
            path=(out/rel).resolve()
            require(out in path.parents and sha256(path)==digest,'committed artifact differs: '+rel)


def selected(seed,r,part,cfg):
    from pcla_repair.core import keyed_seed
    return random.Random(keyed_seed(seed,'participants',r)).sample(sorted(map(int,part['clients'])),cfg['selected_clients'])


def audit_aggregation(out, meta, ev_id, partition, require_complete):
    from pcla_repair import core,runtime as rt
    out=Path(out); run=out/'run'; cp=rt.load_checkpoint(run/'checkpoint.pt')
    require(cp['identity_hash']==rt.fingerprint(cp['identity'])==meta['identity_hash'],'checkpoint identity differs')
    require(cp['policy']==meta['policy'] and cp['identity']==meta['identity'],'checkpoint policy/config differs')
    spec=cp['identity']['provenance']['v41']; cfg=cp['identity']['config'];seed=cp['identity']['seed']
    require(spec['source_files']==source_files(),'execution source differs from audited code')
    n=cp['round']; require(len(cp['metrics'])==n,'checkpoint metrics incomplete')
    require([r['round'] for r in cp['metrics']]==list(range(1,n+1)),'non-contiguous rounds')
    require(read(run/'metrics.json')==cp['metrics'],'published metrics/checkpoint disagree')
    require(set(cp['v41_evidence'])==set(map(str,range(1,n+1))),'missing committed sidecars')
    verify_files(out,cp['v41_evidence'].values())
    old=None; donor=None
    if meta.get('embedded_source_sha256'):
        require(sha256(out/'source_reference.pt')==meta['embedded_source_sha256'],'embedded source changed')
        old=tensor_load(out/'source_reference.pt')
        require(old['identity_hash']==rt.fingerprint(old['identity']),'source reference identity invalid')
        if spec['source_original_identity'] is not None:
            require(old['identity_hash']==spec['source_original_identity'],'source-parent identity differs')
        require(old['original_file_sha256']==spec['source_original_checkpoint_sha256'],'source file binding differs')
    if meta.get('embedded_donor_sha256'):
        require(sha256(out/'donor_reference.pt')==meta['embedded_donor_sha256'],'embedded donor changed')
        donor=tensor_load(out/'donor_reference.pt')
        require(donor['identity_hash']==rt.fingerprint(donor['identity'])==cp['identity_hash'],
                'online donor identity differs')
        require(donor['policy']=='pcla' and donor['round']==cfg['rounds'],'donor incomplete')
        require(donor['original_file_sha256']==meta['online_donor']['sha256'],'donor file binding differs')
    trace_keys=set(); rows=[]; max_age={}; last_covered={}; matched_count=0
    for r,row in enumerate(cp['metrics'],1):
        require(row['selected_clients']==selected(seed,r,partition,cfg),'participant sequence differs')
        clients=read(run/f'clients_round_{r:03d}.json')
        require([c['client'] for c in clients]==row['selected_clients'],'client details order differs')
        factor='b' if r%2 else 'a'; weighted=None; total=0
        for c in clients:
            cid=c['client'];key=f'{r}/{cid}';trace_keys.add(key)
            mask=torch.tensor(c['support'],dtype=torch.bool)
            budget=partition['capacities'][str(cid)]*cfg['credits_per_layer']
            core.check_support(mask,tuple(cp['trace'][key].shape),budget)
            require(torch.equal(mask,cp['trace'][key]),'saved support/trace differs')
            require(c['factor']==factor and c['sample_weight']==len(partition['clients'][str(cid)]),'client factor/weight differs')
            require(c['credits']==budget,'credit count differs')
            if spec['family']=='fixed_support' and old is not None:
                require(torch.equal(mask,old['trace'][key]),'fixed donor support changed')
            if donor is not None:
                require(torch.equal(mask.sum(1),donor['trace'][key].sum(1)),'online donor per-layer quota differs')
            if meta.get('source_policy')=='random':
                expected=core.allocate('random',torch.zeros(mask.shape),partition['capacities'][str(cid)],
                    cfg['credits_per_layer'],core.keyed_seed(seed,'allocation','random',r,cid))
                require(torch.equal(mask,expected),'random support could not be regenerated')
            weighted=mask.double()*c['sample_weight'] if weighted is None else weighted+mask.double()*c['sample_weight']
            total+=c['sample_weight']
        a=read(out/'evidence'/f'{r:03d}'/'aggregation.json')
        require(a['rule']==spec['rule'] and a['factor']==factor,'aggregation treatment differs')
        names=sorted(k[:-7] for k in cp['adapter'] if k.endswith('.lora_a'))
        # Module layer is stored by the native upload layout? Use source model names or support equality per layer.
        coverage=a['coverage_fraction_by_module_rank']
        require(set(names)==set(coverage),'coverage module inventory differs')
        for name,values in coverage.items():
            import re
            match=re.search(r'(?:layer|blocks)\.(\d+)\.',name)
            require(match is not None,'cannot resolve layer identity')
            layer=int(match.group(1))
            require(torch.allclose(torch.tensor(values,dtype=torch.float64),weighted[layer]/total,atol=0,rtol=0),
                    'recorded coverage differs from masks')
            key=(factor,name)
            if key not in last_covered: last_covered[key]=[0]*len(values);max_age[key]=[0]*len(values)
            for j,v in enumerate(values):
                max_age[key][j]=max(max_age[key][j],r-last_covered[key][j])
                if v>0:last_covered[key][j]=r
        require(a['classifier_equal_across_rules'] is True,'classifier comparison missing')
        if spec['rule']=='function_norm_matched' and a['function_match_defined']:
            require(abs(a['requested_effective_norm']-a['contributors_function_norm'])<=
                    1e-10*max(1.,a['contributors_function_norm']),'G did not match effective norm')
            matched_count+=1
        e=read(out/'evidence'/f'{r:03d}'/'evaluation.json');stats=compare_evaluation(e,ev_id['labels'])
        require(stats['predictions']==read(run/f'predictions_round_{r:03d}.json'),'native predictions differ')
        require(row['accuracy']==stats['accuracy'] and row['loss']==e['loss'],'native quality row differs')
        rows.append({'round':r,**{k:v for k,v in stats.items() if k!='predictions'},
                     'function_match_defined':a['function_match_defined'],
                     'train_seconds':row['train_seconds'],'probe_seconds':row['probe_seconds'],
                     'serialized_traffic_bytes':row['serialized_traffic_bytes']})
    require(set(cp['trace'])==trace_keys,'trace has missing or extra entries')
    for r in cfg['probe_rounds']:
        if r<=n:
            probe=rt.load_checkpoint(run/f'probe_round_{r:03d}.pt')
            require(probe['identity_hash']==cp['identity_hash'] and probe['round']==r,'probe identity differs')
            if r==n: require(all(torch.equal(v,probe['adapter'][k]) for k,v in cp['adapter'].items()),'final/probe state differs')
    done=read(out/'status.json')['state']=='completed' and n==cfg['rounds']
    require(done or not require_complete,'run has not completed fixed horizon')
    return {'kind':'aggregation','complete':done,'seed':seed,'family':spec['family'],
            'source_policy':meta['source_policy'],'policy':meta['policy'],'rule':spec['rule'],
            'rows':rows,'function_matching_rounds':matched_count,
            'max_no_coverage_gap_by_factor_module':{str(k):v for k,v in max_age.items()}}


def audit_joint_bank(bank, mode, record):
    from lora_matrix import weighted_effective_mean, exact_server_correction
    base=bank['parent_adapter']; total=sum(c['weight'] for c in bank['clients'])
    mean={n:(v.double()+sum(c['weight']*(c['adapter'][n].double()-v.double()) for c in bank['clients'])/total).to(v.dtype)
          for n,v in base.items()}
    require([c['client'] for c in bank['clients']]==record['selected_clients'],'joint bank clients differ')
    expected={}
    for name,scale in bank['scalings'].items():
        entries=[(c['weight'],scale,c['adapter'][name+'.lora_b'],c['adapter'][name+'.lora_a']) for c in bank['clients']]
        target=weighted_effective_mean(entries)
        _,diag=exact_server_correction(bank['parent_projections'][name],mean[name+'.lora_b'],mean[name+'.lora_a'],scale,target)
        expected[name]=diag
    require(len(record['algebra'])==len(expected),'joint algebra inventory differs')
    for x in record['algebra']:
        y=expected[x['module']]
        for field in ('target_norm','residual_norm','deployed_error_max_abs'):
            require(abs(x[field]-y[field])<=1e-10*max(1.,abs(y[field])),'joint residual audit differs')
    return True


def client_labels(part,ev_id,repo):
    if ev_id['synthetic']:
        from pcla_repair.testing import TinyDataset
        return {s:int(y) for c,ids in part['clients'].items() for s,y in zip(ids,TinyDataset(100+int(c),len(ids)).y.tolist())}
    flat=[s for ids in part['clients'].values() for s in ids]
    return dict(zip(flat,labels_from_tsv(repo,{'all_clients':flat},ev_id['data_provenance'],'all_clients')))


def audit_iid_round(record,seed,r,part,cfg,participants,labels):
    from pcla_repair.core import keyed_seed
    population=[(c,s) for c in sorted(map(int,part['clients'])) for s in part['clients'][str(c)]]
    n=sum(len(part['clients'][str(c)]) for c in participants)
    drawn=[population[i] for i in sorted(random.Random(keyed_seed(seed,'iid_round',r)).sample(range(len(population)),n))]
    cohort=[s for c in participants for s in part['clients'][str(c)]]
    orders=[torch.randperm(n,generator=torch.Generator().manual_seed(keyed_seed(seed,'central_order',r,e))).tolist()
            for e in range(cfg['local_epochs'])]
    require(record['cohort_sample_multiset_sha256']==fingerprint(cohort),'IID cohort reference differs')
    require(record['sample_multiset_sha256']==fingerprint([s for _,s in drawn]) and record['iid_sample_count']==n,
            'IID population draw differs')
    require(record['pooled_order_sha256']==fingerprint(orders),'IID pooled order differs')
    require(record['iid_positive_fraction']==sum(labels[s] for _,s in drawn)/n and
            record['cohort_positive_fraction']==sum(labels[s] for s in cohort)/len(cohort),'IID label readout differs')
    require(record['iid_samples_from_cohort']==sum(c in participants for c,_ in drawn) and
            record['iid_clients_represented']==len({c for c,_ in drawn}),'IID client readout differs')


def audit_ffa_bank(bank,record,part,reference_a,probe,lr=1.0):
    require(bank['factor']=='b' and [c['client'] for c in bank['clients']]==record['selected_clients'],'FFA bank treatment differs')
    require(all(c['weight']==len(part['clients'][str(c['client'])]) for c in bank['clients']),'FFA client weight differs')
    for state in [bank['parent_adapter']]+[c['adapter'] for c in bank['clients']]:
        require(all(torch.equal(state[n],v) for n,v in reference_a.items()),'FFA bank A differs from shared initial A')
    from pcla_repair.core import tensor_digest
    base=bank['parent_adapter'];total=sum(c['weight'] for c in bank['clients'])
    require(tensor_digest(base)==record['initial_round_adapter_sha256'],'FFA bank parent differs from round start')
    mean=bank_mean(base,bank['clients'],lr)
    require(tensor_digest(mean)==record['post_round_adapter_sha256'],'FFA B/head aggregate differs from fixed-denominator FedAvg')
    if probe is not None:require(all(torch.equal(mean[n],v) for n,v in probe.items()),'FFA probe differs from aggregate')


def bank_mean(parent,clients,lr=1.0):
    total=sum(c['weight'] for c in clients);out={}
    for n,v in parent.items():
        s=sum(c['weight']*(c['adapter'][n].double()-v.double()) for c in clients)
        out[n]=(v.double()+s/total if lr==1 else v.double()+lr*(s/total)).to(v.dtype)
    return out


def module_layers(names):
    import re
    layers={}
    for n in names:
        match=re.search(r'(?:layer|blocks)\.(\d+)\.',n);require(match is not None,'cannot resolve layer identity')
        layers[n]=int(match.group(1))
    return layers


def svd_sign_ok(b,values):
    top=float(values[0]) if len(values) else 0.
    return all(float(b[:,j].max())>=float(-b[:,j].min())*(1-1e-6) for j in range(b.shape[1]) if float(values[j])>1e-9*top)


def check_params(identity,cfg):
    from run_lora_controls import MODES,SERVER_LR_MODES,EMA_MODES,COHORTS
    mode,p=identity['mode'],identity['params']
    require(set(p)=={'rounds','server_lr','cohort','ema_readout'} and p['rounds']==cfg['rounds'],'run parameters differ from config')
    require(mode in MODES and p['cohort'] in COHORTS and math.isfinite(p['server_lr']) and p['server_lr']>0,'unknown mode/cohort/server_lr')
    require(p['server_lr']==1 or mode in SERVER_LR_MODES,'server_lr outside base-invariant federated means')
    require(p['ema_readout'] is None or (mode in EMA_MODES and 0<p['ema_readout']<1),'EMA readout outside allowed modes')
    require(mode!='bridge_full_alt' or (p['server_lr'],p['cohort'],p['ema_readout'])==(1,'uniform',None),'bridge parameters differ')
    require(mode!='local_only' or p['cohort']=='uniform','local_only cohort differs')
    return p


def audit_strata(identity,part,labels,cfg):
    fractions={int(c):sum(labels[s] for s in ids)/len(ids) for c,ids in part['clients'].items()}
    order=sorted(fractions,key=lambda c:(fractions[c],c));k=cfg['selected_clients']
    strata=[order[i*len(order)//k:(i+1)*len(order)//k] for i in range(k)]
    require(identity['strata']==strata and identity['client_positive_fraction']=={str(c):v for c,v in fractions.items()},
            'label strata could not be recomputed')
    return strata,fractions


def participants(seed,r,part,cfg,strata):
    if strata is None:return selected(seed,r,part,cfg)
    from pcla_repair.core import keyed_seed
    rng=random.Random(keyed_seed(seed,'stratified_participants',r))
    chosen=[rng.choice(g) for g in strata]
    require([sum(c in g for c in chosen) for g in strata]==[1]*len(strata),'stratified cohort is not one client per stratum')
    return chosen


def majority_readout(train,test):
    guess=int(2*sum(train)>=len(train));present=sorted(set(test))
    recalls=[sum(1 for y in test if y==c and y==guess)/sum(1 for y in test if y==c) for c in present]
    return {'guess':guess,'accuracy':sum(1 for y in test if y==guess)/len(test),'balanced_accuracy':sum(recalls)/len(recalls)}


def audit_personal(path,mode,seed,ids,labels,train):
    pe=read(path)
    require(pe['mode']==mode and pe['seed']==seed and pe['test_identity_sha256']==fingerprint([ids,labels]),'personal evaluation identity differs')
    require(set(pe['clients'])==set(ids),'personal evaluation client inventory differs')
    rows=[]
    for k in sorted(ids,key=int):
        x=pe['clients'][k];stats=compare_evaluation(x['evaluation'],labels[k])
        require(x['source_ids']==ids[k] and x['client']==int(k),'personal test IDs differ')
        require(x['accuracy']==stats['accuracy'] and abs(x['balanced_accuracy']-stats['balanced_accuracy'])<1e-14,'personal metric differs from logits')
        m=majority_readout(train[int(k)],labels[k])
        require(x['majority']['guess']==m['guess'] and x['majority']['accuracy']==m['accuracy'] and
                abs(x['majority']['balanced_accuracy']-m['balanced_accuracy'])<1e-14,'majority baseline differs')
        rows.append({'client':int(k),**{f:v for f,v in stats.items() if f!='predictions'},'majority':m})
    mean=lambda f:sum(f(x) for x in rows)/len(rows)
    summary={'clients':len(rows),'mean_accuracy':mean(lambda x:x['accuracy']),'mean_balanced_accuracy':mean(lambda x:x['balanced_accuracy']),
             'mean_majority_accuracy':mean(lambda x:x['majority']['accuracy']),
             'mean_majority_balanced_accuracy':mean(lambda x:x['majority']['balanced_accuracy'])}
    require(all(abs(pe['summary'][k]-v)<1e-12 for k,v in summary.items()),'personal summary differs')
    return {'summary':summary,'clients':rows}


def client_train_labels(part,labels):
    return {int(c):[labels[s] for s in ids] for c,ids in part['clients'].items()}


def audit_ema(out,records,params,cp,ev_id):
    from pcla_repair.core import tensor_digest
    beta=params['ema_readout'];previous=None
    for i,rec in enumerate(records):
        r=rec['round'];e=rec['ema_readout']
        require(e['beta']==beta and e['initialization']=='first_post_aggregation_state' and e['previous_ema_sha256']==previous,'EMA chain differs')
        compare_evaluation(rec['ema_evaluation'],ev_id['labels'])
        nxt=records[i+1]['initial_round_adapter_sha256'] if i+1<len(records) else tensor_digest(cp['adapter'])
        require(e['global_sha256']==nxt,'EMA global state is not the training state')
        anchor=out/'rounds'/f'{r:03d}'/'ema_anchor.pt'
        if anchor.exists():
            x=tensor_load(anchor);cur=x['current_global']
            require(tensor_digest(cur)==e['global_sha256'],'EMA anchor global differs')
            if x['previous'] is None:
                require(r==1,'EMA initialization after round 1');expected={n:v.double() for n,v in cur.items()}
            else:
                require(tensor_digest(x['previous'])==previous,'EMA anchor previous differs')
                expected={n:beta*x['previous'][n]+(1-beta)*v.double() for n,v in cur.items()}
            require(set(expected)==set(x['ema']) and all(torch.equal(v,x['ema'][n]) for n,v in expected.items()) and
                    tensor_digest(x['ema'])==e['ema_sha256'],'EMA state could not be recomputed at anchor')
        previous=e['ema_sha256']
    require(records[0]['round']>1 or (out/'rounds/001/ema_anchor.pt').exists(),'EMA initialization anchor missing')
    require(tensor_digest(cp['ema'])==previous,'checkpoint EMA differs from last round')


def audit_svd_bank(bank,record,probe,mode):
    from lora_matrix import weighted_effective_mean
    from pcla_repair.core import tensor_digest
    require(tensor_digest(probe['adapter'])==record['post_round_adapter_sha256'] and
            tensor_digest(probe['projections'])==record['post_round_base_sha256'],'SVD anchor probe differs from record')
    mean=bank_mean(bank['parent_adapter'],bank['clients']);eps=torch.finfo(torch.float32).eps
    require([x['module'] for x in record['svd']]==list(bank['scalings']),'SVD module inventory differs')
    for x in record['svd']:
        n=x['module'];s=bank['scalings'][n]
        m=mean[n+'.lora_b'].double()@mean[n+'.lora_a'].double();r=mean[n+'.lora_a'].shape[0]
        values=torch.linalg.svdvals(m)[:r];top=float(values[0]);tol=1e-5*top
        require(max(abs(u-v) for u,v in zip(x['singular_values'],values.tolist()))<=1e-9*max(top,1e-300),'recorded singular values differ')
        nb,na=probe['adapter'][n+'.lora_b'].double(),probe['adapter'][n+'.lora_a'].double()
        require(float((nb@na-m).abs().max())<=tol,'SVD refactor changed the LoRA product')
        require(float((nb.T@nb-torch.diag(values)).abs().max())<=tol and float((na@na.T-torch.diag(values)).abs().max())<=tol,
                'refactored factors are not balanced top-r SVD factors')
        require(svd_sign_ok(nb,values),'SVD sign convention violated')
        target=(weighted_effective_mean([(c['weight'],s,c['adapter'][n+'.lora_b'],c['adapter'][n+'.lora_a']) for c in bank['clients']])
                if mode=='fed_svd_refactor' else s*m)
        desired=bank['parent_projections'][n].double()+target
        achieved=probe['projections'][n].double()+s*(nb@na)
        require(float((achieved-desired).abs().max())<=4*eps*max(1.,float(desired.abs().max())),'SVD refactor effective weight not exact')


def audit_svd_basis(out,identity,initial):
    from pcla_repair.core import tensor_digest
    info=identity['svd_basis'];rand=tensor_load(out/'random_initial_adapter.pt')
    require(tensor_digest(rand)==info['random_initial_adapter_sha256']!=identity['initial_adapter_sha256'],'random initial adapter differs')
    require(all(torch.equal(v,rand[n]) for n,v in initial.items() if not n.endswith('.lora_a')),'SVD basis changed B/head')
    projections=tensor_load(out/'checkpoint.pt')['projections']
    require(set(info['modules'])==set(projections),'SVD basis inventory differs')
    for n,x in info['modules'].items():
        u,values,vh=torch.linalg.svd(projections[n].double(),full_matrices=False);r=initial[n+'.lora_a'].shape[0]
        u,vh=u[:,:r].clone(),vh[:r].clone()
        for j in range(r):
            if u[int(u[:,j].abs().argmax()),j]<0:u[:,j]*=-1;vh[j]*=-1
        norm=rand[n+'.lora_a'].double().norm();expected=vh*(norm/vh.norm());a=initial[n+'.lora_a'].double()
        require(abs(float(norm)-x['random_a_frobenius_norm'])<=1e-12*float(norm) and abs(float(a.norm())-float(norm))<=1e-5*float(norm),
                'SVD basis norm differs from random A')
        require(float((a-expected).abs().max())<=1e-6*float(expected.abs().max()),'frozen A is not the top-r W0 right singular basis')
        require(max(abs(p-q) for p,q in zip(x['w0_top_singular_values'],values[:r].tolist()))<=1e-9*float(values[0]),'W0 singular values differ')


def audit_central(out,records,identity,cfg,seed,synthetic,cp):
    from pcla_repair import core
    mode=identity['mode'];b=identity['central_budget']
    layers=len(records[0]['support']);rank=len(records[0]['support'][0])
    q,m=(layers,rank) if mode=='central_full_alt' else (2 if synthetic else 8,cfg['credits_per_layer'])
    require(b=={'layers':q,'credits_per_layer':m,'components':q*m,
                'selection':{'central_full_alt':'all','central_pcla':'pcla','central_random_rank':'random',
                             'central_replay_random_rank':'matched_replay_random_rank'}[mode]},'central budget differs')
    require(identity.get('optimizer_scope')=='per_round' and cp['optimizer'] is None,'central optimizer scope differs')
    states={'a':core.UtilityState(),'b':core.UtilityState()}
    matched = read(out/'matched_donor.json') if mode=='central_replay_random_rank' else None
    if matched is not None:
        require(set(matched)==set(map(str,range(1,len(records)+1))), 'central donor round inventory differs')
        require(identity['matched_donor']['record_sha256']==
                {r:v['record_sha256'] for r,v in matched.items()}, 'central donor hashes differ')
    for i,rec in enumerate(records):
        r=rec['round'];factor='b' if r%2 else 'a';support=torch.tensor(rec['support'],dtype=torch.bool)
        core.check_support(support,(layers,rank),q*m)
        require(rec['factor']==factor and rec['optimizer_scope']=='per_round' and int(support.any(1).sum())==q and rec['central_budget']=={'layers':q,'components':q*m},
                'central factor/budget differs')
        if mode=='central_full_alt':require(rec['selection']=={},'full central selection differs')
        elif mode=='central_random_rank':
            expected=core.allocate('random',torch.zeros(layers,rank,dtype=torch.float64),q,m,core.keyed_seed(seed,'allocation','random',r,'central'))
            require(torch.equal(support,expected) and rec['selection']=={},'random central support could not be regenerated')
        elif mode=='central_replay_random_rank':
            donor=matched[str(r)];mask=torch.tensor(donor['support'],dtype=torch.bool)
            core.check_support(mask,(layers,rank),q*m)
            expected=core.allocate('replay_random_rank',torch.zeros(layers,rank,dtype=torch.float64),q,m,
                                   core.keyed_seed(seed,'allocation','replay_random_rank',r,'central'),donor=mask)
            require(torch.equal(expected,support) and torch.equal(support.sum(1),mask.sum(1)),
                    'matched central layer quotas or random rank allocation differ')
            require(rec['selection']=={'donor_record_sha256':donor['record_sha256'],
                                      'donor_layer_quotas':mask.sum(1).tolist()},
                    'matched central donor record differs')
            for key in ('selected_clients','sample_multiset_sha256','pooled_order_sha256'):
                require(rec[key]==donor[key], 'matched central training exposure differs: '+key)
        else:
            st=states[factor];sel=rec['selection'];due=st.importance is None or r-st.last_refresh>=cfg['refresh_rounds']
            require(sel['score_refreshed']==due and (sel['raw_probe_scores'] is not None)==due,'PCLA refresh schedule differs')
            if due:st.refresh(torch.tensor(sel['raw_probe_scores'],dtype=torch.float64),r,cfg['ema'])
            scores=st.score(r,cfg['beta'],cfg['gamma'])
            require(scores.tolist()==sel['scores'],'PCLA utility scores could not be replayed')
            expected=core.allocate('pcla',scores,q,m,core.keyed_seed(seed,'allocation','pcla',r,'central'),visit=st.visits)
            require(torch.equal(expected,support),'PCLA support could not be regenerated from scores');st.observe(support,r)
        bank=out/'rounds'/f'{r:03d}'/'central_bank.pt'
        if bank.exists():
            x=tensor_load(bank);probe=tensor_load(out/f'probe_round_{r:03d}.pt');after=probe['adapter'];parent=x['parent_adapter']
            nxt=records[i+1]['initial_round_adapter_sha256'] if i+1<len(records) else core.tensor_digest(cp['adapter'])
            require(core.tensor_digest(after)==nxt and core.tensor_digest(parent)==rec['initial_round_adapter_sha256'],'central anchor chain differs')
            require(x['factor']==factor and torch.equal(x['support'],support) and x['layout']==module_layers(x['layout']),'central bank differs')
            partner='a' if factor=='b' else 'b'
            for n,l in x['layout'].items():
                key=n+'.lora_'+factor;keep=support[l][:,None] if factor=='a' else support[l][None,:];keep=keep.expand_as(after[key])
                require(torch.equal(after[n+'.lora_'+partner],parent[n+'.lora_'+partner]) and torch.equal(after[key][~keep],parent[key][~keep]),
                        'central unselected or inactive-factor entries changed')
            audit_fresh_optimizer(x,support,factor,rec['steps'])
    if mode=='central_pcla':
        saved={k:core.UtilityState(**v) for k,v in cp['utility_states'].items()}
        for k,st in states.items():
            require(all((torch.equal(getattr(st,f),getattr(saved[k],f)) if torch.is_tensor(getattr(st,f)) else getattr(st,f)==getattr(saved[k],f))
                        for f in ('importance','counts','last_selected','last_refresh','visits')),'checkpoint utility state differs from replay')


def audit_fresh_optimizer(bank,support,factor,steps):
    """A new AdamW per round: only this round's trainable tensors, step==round steps, unselected moments zero."""
    names,state=bank['optimizer_names'],bank['round_optimizer_state']
    trainable=[n+'.lora_'+factor for n,l in bank['layout'].items() if bool(support[l].any())]+[n for n in bank['parent_adapter'] if n.startswith('classifier.')]
    require(sorted(names)==sorted(trainable) and set(state)==set(range(len(names))),'round optimizer inventory differs')
    for i,n in enumerate(names):
        require(int(state[i]['step'])==steps,'central optimizer did not start fresh this round')
        if n.startswith('classifier.'):continue
        row=support[bank['layout'][n.rsplit('.',1)[0]]];shape=bank['parent_adapter'][n].shape
        frozen=~(row[:,None] if factor=='a' else row[None,:]).expand(shape)
        require(all(not bool(state[i][k][frozen].any()) for k in ('exp_avg','exp_avg_sq')),'unselected central moments are not zero')


def audit_fed_sa(out,records,identity,part,cp):
    from pcla_repair.core import tensor_digest
    clients=sorted(map(int,part['clients']));after={c:identity['initial_personal_b_sha256'] for c in clients};shared_prev=None
    for rec in records:
        r=rec['round'];sel=rec['selected_clients']
        require(rec['personal_b_before_sha256']=={str(c):after[c] for c in sel},'client did not start from its own personal B')
        now={int(c):h for c,h in rec['personal_b_after_sha256'].items()}
        require(set(now)==set(clients) and all(now[c]==after[c] for c in clients if c not in sel),'non-participant personal B changed')
        require(shared_prev is None or rec['initial_round_shared_sha256']==shared_prev,'shared A/head chain broken')
        bank=out/'rounds'/f'{r:03d}'/'fed_sa_bank.pt'
        if bank.exists():
            x=tensor_load(bank);probe=tensor_load(out/f'probe_round_{r:03d}.pt')
            require([c['client'] for c in x['clients']]==sel and all(c['weight']==len(part['clients'][str(c['client'])]) for c in x['clients']),
                    'FedSA bank clients differ')
            require(tensor_digest(x['parent_shared'])==rec['initial_round_shared_sha256'],'FedSA bank parent differs')
            if r==1:
                any_b=x['parent_personal_b'][str(sel[0])]
                require(tensor_digest({**x['parent_shared'],**any_b})==identity['initial_adapter_sha256'],'FedSA initial state differs')
            for c in x['clients']:
                cid=str(c['client']);b={n:v for n,v in c['adapter'].items() if n.endswith('.lora_b')}
                require(tensor_digest(x['parent_personal_b'][cid])==rec['personal_b_before_sha256'][cid] and
                        tensor_digest(b)==rec['personal_b_after_sha256'][cid],'FedSA personal B in bank differs')
            shared=bank_mean(x['parent_shared'],[{'weight':c['weight'],'adapter':{n:c['adapter'][n] for n in x['parent_shared']}} for c in x['clients']])
            require(tensor_digest(shared)==rec['post_round_shared_sha256'],'FedSA shared A/head is not the fixed-denominator mean')
            personal=probe['personal_b'];ids=sorted(personal,key=int)
            require([tensor_digest(personal[c]) for c in ids]==[rec['personal_b_after_sha256'][c] for c in ids],'FedSA probe personal B differs')
            mean_b={k:(sum(personal[c][k].double() for c in ids)/len(ids)).to(v.dtype) for k,v in personal[ids[0]].items()}
            require(tensor_digest(mean_b)==rec['mean_personal_b_sha256'] and tensor_digest(probe['adapter'])==tensor_digest({**shared,**mean_b}),
                    'FedSA evaluation state is not shared A/head + mean personal B')
        after=now;shared_prev=rec['post_round_shared_sha256']
    final=cp['personal_b'];adapter=cp['adapter']
    require({int(c):tensor_digest(v) for c,v in final.items()}==after,'checkpoint personal B differs from last round')
    require(tensor_digest({n:v for n,v in adapter.items() if not n.endswith('.lora_b')})==shared_prev and
            tensor_digest({n:v for n,v in adapter.items() if n.endswith('.lora_b')})==records[-1]['mean_personal_b_sha256'],
            'checkpoint FedSA state differs from last round')


def audit_local_only(out,cp,identity,part,ev_id,repo,require_complete,labels):
    from context import expected_local_tests
    from run_lora_controls import LOCAL_ONLY_EPOCHS
    from pcla_repair.core import tensor_digest
    seed=identity['seed'];cfg=identity['config'];order=sorted(map(int,part['clients']))
    ids,test_labels=expected_local_tests(part,seed,ev_id['synthetic'],repo);train=client_train_labels(part,labels)
    require(identity['client_order']==order and identity['local_only_epochs']==LOCAL_ONLY_EPOCHS and
            identity['local_test_sha256']==fingerprint([ids,test_labels]),'local_only identity differs')
    require(tensor_digest(cp['adapter'])==identity['initial_adapter_sha256'] and
            tensor_digest(cp['projections'])==identity['initial_projections_sha256'],'local_only shared initial state changed')
    rows=[]
    for row in cp['metrics']:
        i=row['round'];cid=order[i-1];x=read(out/'clients'/f'{cid:03d}.json');n=len(part['clients'][str(cid)])
        require(row['client']==cid and x['client']==cid and x['ordinal']==i and x['epochs']==LOCAL_ONLY_EPOCHS and
                x['steps']==math.ceil(n/cfg['batch_size'])*LOCAL_ONLY_EPOCHS and x['examples']==n*LOCAL_ONLY_EPOCHS and
                x['initial_adapter_sha256']==identity['initial_adapter_sha256'],'local_only client record differs')
        stats=compare_evaluation(x['evaluation'],test_labels[str(cid)])
        require(x['source_ids']==ids[str(cid)] and row['accuracy']==x['accuracy']==stats['accuracy'] and row['gpu_nll']==stats['gpu_nll'],
                'local_only metric differs from logits')
        rows.append({'round':i,'client':cid,**{k:v for k,v in stats.items() if k!='predictions'},'train_seconds':x['train_seconds']})
    complete=len(rows)==len(order);personal=None
    if complete:
        personal=audit_personal(out/'personal_evaluation.json','local_only',seed,ids,test_labels,train)
        pe=read(out/'personal_evaluation.json')
        require(all(pe['clients'][str(r['client'])]['evaluation']==read(out/'clients'/f"{r['client']:03d}.json")['evaluation'] for r in rows),
                'personal evaluation differs from committed client records')
    done=read(out/'status.json')['state']=='completed' and complete
    require(done or not require_complete,'local_only incomplete')
    return {'kind':'lora_control','complete':done,'seed':seed,'mode':'local_only','params':identity['params'],'rows':rows,'personal':personal}


def audit_control(out,meta,ev_id,part,require_complete,repo=None):
    from run_lora_controls import (FFA_FIDELITY,FFA_SVD_FIDELITY,FFA,FED,IID,CENTRAL_ALT,SVD_REFACTOR,BASE_CHANGING)
    from pcla_repair.core import tensor_digest
    out=Path(out);cp=tensor_load(out/'checkpoint.pt');identity=cp['identity'];cfg=identity['config'];seed=identity['seed']
    require(fingerprint(identity)==cp['identity_hash']==meta['identity_hash'],'control identity differs')
    require(identity==meta['identity'] and identity['source_files']==source_files(),'control source differs')
    verify_files(out,[row['evidence'] for row in cp['metrics']])
    require([x['round'] for x in cp['metrics']]==list(range(1,cp['round']+1)),'control rounds incomplete')
    mode=identity['mode'];params=check_params(identity,cfg);rows=[]
    need=mode in IID or mode in ('fed_sa','local_only') or params['cohort']=='label_stratified'
    labels=client_labels(part,ev_id,repo) if need else None
    if mode=='local_only':return audit_local_only(out,cp,identity,part,ev_id,repo,require_complete,labels)
    strata,fractions=audit_strata(identity,part,labels,cfg) if params['cohort']=='label_stratified' else (None,None)
    reference_a=None;previous=identity['initial_adapter_sha256'];lr=params['server_lr']
    records=[read(out/'rounds'/f"{row['round']:03d}"/'record.json') for row in cp['metrics']]
    if mode in FFA:
        first=tensor_load(out/'rounds/001/local_bank.pt')['parent_adapter']
        require(tensor_digest(first)==identity['initial_adapter_sha256'],'FFA initial adapter differs')
        reference_a={n:v for n,v in first.items() if n.endswith('.lora_a')}
        require(all(torch.equal(cp['adapter'][n],v) for n,v in reference_a.items()),'FFA final A differs from shared initial A')
        if mode=='ffa_svd_basis':audit_svd_basis(out,identity,first)
    for i,(row,record) in enumerate(zip(cp['metrics'],records)):
        r=row['round']
        expected=participants(seed,r,part,cfg,strata)
        require(record['mode']==mode and record['selected_clients']==expected,'control treatment/participants differ')
        if strata is not None:
            require(record['cohort']=='label_stratified' and record['participant_positive_fraction']==[fractions[c] for c in expected],
                    'stratified participant label readout differs')
        if mode in IID:audit_iid_round(record,seed,r,part,cfg,expected,labels)
        else:require(record['sample_multiset_sha256']==fingerprint([s for c in expected for s in part['clients'][str(c)]]),
                     'control sample exposure differs')
        nxt=records[i+1]['initial_round_adapter_sha256'] if i+1<len(records) else tensor_digest(cp['adapter'])
        if mode in FFA:
            require(record['factor']=='b' and all(c['factor']=='b' for c in record['client_details']),'FFA trained factor differs')
            require(record['lora_a_sha256']==tensor_digest(reference_a) and
                    record['fidelity']==(FFA_FIDELITY if mode=='fed_ffa_b' else FFA_SVD_FIDELITY),'FFA record differs')
            require(record['initial_round_adapter_sha256']==previous,'FFA round chain broken')
            previous=record['post_round_adapter_sha256']
            require(all(x['residual_norm']<1e-4*max(1.,x['target_norm']) for x in record['algebra']),'FFA factor mean is not exact')
        if mode in FED and lr!=1:
            require(record['server_lr']==lr and record['post_round_adapter_sha256']==nxt,'server-lr record/chain differs')
        if mode in SVD_REFACTOR:
            require(record['post_round_adapter_sha256']==nxt and len(record['svd'])==len(record['algebra']),'SVD record/chain differs')
            for x in record['svd']:
                v=x['singular_values']
                require(all(math.isfinite(y) and y>=0 for y in v) and v==sorted(v,reverse=True) and math.isfinite(x['refactor_residual_norm']),
                        'SVD singular values invalid')
        expected_steps=sum(math.ceil(len(part['clients'][str(c)])/cfg['batch_size']) for c in expected)*cfg['local_epochs']
        expected_samples=sum(len(part['clients'][str(c)]) for c in expected)*cfg['local_epochs']
        require(record['steps']==expected_steps and record['examples']==expected_samples,'control budget differs')
        stats=compare_evaluation(record['evaluation'],ev_id['labels'])
        require(row['accuracy']==stats['accuracy'] and row['gpu_nll']==stats['gpu_nll'],'control row differs')
        if 'same_update_alternative' in record:
            compare_evaluation(record['same_update_alternative']['evaluation'],ev_id['labels'])
        bank=out/'rounds'/f'{r:03d}'/'local_bank.pt'
        if bank.exists():
            require(mode in FED,'unexpected local bank')
            b=tensor_load(bank);audit_joint_bank(b,identity['mode'],record)
            if mode in FFA:
                probe=out/f'probe_round_{r:03d}.pt'
                audit_ffa_bank(b,record,part,reference_a,tensor_load(probe)['adapter'] if probe.exists() else None,lr)
            if lr!=1:
                require(tensor_digest(bank_mean(b['parent_adapter'],b['clients']))==record['aggregate_mean_sha256'] and
                        tensor_digest(bank_mean(b['parent_adapter'],b['clients'],lr))==record['post_round_adapter_sha256'],
                        'server-lr aggregate could not be recomputed from bank')
            if mode in SVD_REFACTOR:audit_svd_bank(b,record,tensor_load(out/f'probe_round_{r:03d}.pt'),mode)
        stat_row={'round':r,**{k:v for k,v in stats.items() if k!='predictions'},
                  'train_and_server_seconds':record['training_and_server_seconds'],
                  'upload_bytes':record['upload_bytes'],'download_bytes':record['download_bytes'],
                  **{k:record[k] for k in ('iid_positive_fraction','cohort_positive_fraction','download_bytes_b_head_only') if k in record}}
        if 'ema_evaluation' in record:
            e=classification(record['ema_evaluation']['logits'],ev_id['labels'])
            stat_row.update({'ema_accuracy':e['accuracy'],'ema_balanced_accuracy':e['balanced_accuracy'],
                             'ema_nll_float64':e['nll_float64'],'ema_gpu_nll':record['ema_evaluation']['loss']})
        if strata is not None:stat_row['participant_positive_fraction']=record['participant_positive_fraction']
        rows.append(stat_row)
    if params['ema_readout'] is not None:audit_ema(out,records,params,cp,ev_id)
    else:require(all('ema_readout' not in x and 'ema_evaluation' not in x for x in records),'unexpected EMA readout')
    if mode in CENTRAL_ALT:audit_central(out,records,identity,cfg,seed,ev_id['synthetic'],cp)
    if mode=='fed_sa':audit_fed_sa(out,records,identity,part,cp)
    if mode in SVD_REFACTOR:
        for n in (k[:-7] for k in cp['adapter'] if k.endswith('.lora_b')):
            b,a=cp['adapter'][n+'.lora_b'].double(),cp['adapter'][n+'.lora_a'].double();g=b.T@b
            values=torch.diagonal(g);tol=1e-5*max(float(values.max()),1e-300)
            require(float((g-torch.diag(values)).abs().max())<=tol and float((a@a.T-g).abs().max())<=tol and svd_sign_ok(b,values),
                    'final SVD factors are not balanced/sign-normalized')
    if mode in FFA:require(previous==tensor_digest(cp['adapter']),'FFA final adapter differs from last round')
    if identity['mode'] not in BASE_CHANGING:
        require(tensor_digest(cp['projections'])==identity['initial_projections_sha256'],'frozen base changed')
    last=out/f"probe_round_{cp['round']:03d}.pt"
    if last.exists():
        snap=tensor_load(last)
        require(snap['identity_hash']==cp['identity_hash'],'last probe identity differs')
        require(tensor_digest(cp['adapter'])==tensor_digest(snap['adapter']) and
                tensor_digest(cp['projections'])==tensor_digest(snap['projections']),'last checkpoint/probe differ')
    done=read(out/'status.json')['state']=='completed' and cp['round']==cfg['rounds']
    require(done or not require_complete,'control horizon incomplete')
    report={'kind':'lora_control','complete':done,'seed':seed,'mode':identity['mode'],'params':params,'rows':rows}
    if mode=='fed_sa':
        pe=out/'personal_evaluation.json'
        if cp['round']==cfg['rounds']:
            require('personal_evaluation.json' in cp['metrics'][-1]['evidence'],'personal evaluation not committed')
            from context import expected_local_tests
            ids,test_labels=expected_local_tests(part,seed,ev_id['synthetic'],repo)
            require(identity['local_test_sha256']==fingerprint([ids,test_labels]),'FedSA local test identity differs')
            report['personal']=audit_personal(pe,'fed_sa',seed,ids,test_labels,client_train_labels(part,labels))
        else:require(not pe.exists(),'personal evaluation before the final round')
    return report


def audit_fixed_state(out,meta,ev_id,part,require_complete,repo=None):
    from aggregation import AggregationSnapshot, compare_updates, RULES
    from run_fixed_state import coverage_table
    out=Path(out);cp=tensor_load(out/'checkpoint.pt');ident=cp['identity']
    require(cp['identity_hash']==fingerprint(ident)==meta['identity_hash'],'G0 identity differs')
    require(ident['source_files']==source_files(),'G0 execution source differs')
    verify_files(out,cp['commits']);rows=[]
    from pcla_repair import core
    support_ref=tensor_load(out/'support_reference.pt')
    require(core.tensor_digest(support_ref['pcla'])==ident['pcla_trace_sha256'] and
            core.tensor_digest(support_ref['replay_random_rank'])==ident['replay_trace_sha256'],
            'G0 source support references differ')
    positive_counts={}
    if ev_id['synthetic']:
        from pcla_repair.testing import TinyDataset
        positive_counts={int(c):int(TinyDataset(100+int(c),len(ids)).y.sum()) for c,ids in part['clients'].items()}
    else:
        flat=[s for ids in part['clients'].values() for s in ids]
        labels=labels_from_tsv(repo,{'all_clients':flat},ev_id['data_provenance'],'all_clients')
        by_source=dict(zip(flat,labels))
        positive_counts={int(c):sum(by_source[s] for s in ids) for c,ids in part['clients'].items()}
    anchor=ident['anchor']
    for i in range(cp['completed_steps']):
        r=anchor+i+1;d=out/'rounds'/f'{r:03d}';p=tensor_load(d/'parent.pt');parent=p['adapter'];snaps={}
        require(p['round']==r and p['factor']==('b' if r%2 else 'a'),'G0 round/factor differs')
        if i==0:require(core.tensor_digest(parent)==ident['parent_sha256'],'G0 initial anchor differs')
        expected_participants=selected(ident['seed'],r,part,meta['identity']['config'])
        for policy in ('pcla','replay_random_rank','full'):
            b=tensor_load(d/(policy+'_local.pt'));bank=b['clients'];total=sum(c['weight'] for c in bank)
            require([c['client'] for c in bank]==expected_participants,'G0 participants differ')
            require(len({c['client'] for c in bank})==len(bank),'duplicate G0 client')
            snap=AggregationSnapshot(parent,total,p['scalings'],p['factor'])
            for c in bank:
                require(c['source_ids']==part['clients'][str(c['client'])] and c['weight']==len(c['source_ids']),
                        'G0 client source differs')
                require(c['capacity']==part['capacities'][str(c['client'])] and
                        c['positive_samples']==positive_counts[c['client']],
                        'G0 contributor labels/capacity do not match independent source')
                if policy=='full':require(bool(c['support'].all()),'full-support reference is sparse')
                else:require(torch.equal(c['support'],support_ref[policy][f"{r}/{c['client']}"]),
                             'G0 selected support differs from its source')
                require(c['optimizer_steps']==math.ceil(c['weight']/ident['config']['batch_size'])*ident['config']['local_epochs'] and
                        c['training_examples_seen']==c['weight']*ident['config']['local_epochs'],
                        'G0 optimizer or sample exposure differs')
                local={n:v.clone() for n,v in parent.items()};local.update(c['active_state']);masks={}
                for n in c['active_state']:
                    if n.startswith('classifier.'):mask=torch.ones_like(local[n],dtype=torch.bool)
                    else:
                        module=n.rsplit('.',1)[0]; v=c['support'][b['layout'][module]]
                        mask=(v[:,None] if p['factor']=='a' else v[None,:]).expand_as(local[n])
                    masks[n]=mask
                snap.add(local,masks,c['weight'],c['client'])
            require(read(d/(policy+'_coverage.json'))==coverage_table(bank,total,tuple(b['shape'])),
                    'coverage/contributor report differs from raw bank')
            snaps[policy]=snap
        ref=snaps['full'].increments()[0]['fixed']
        for policy,snap in snaps.items():
            increments,_=snap.increments()
            for rule in (('fixed',) if policy=='full' else RULES):
                rec=read(d/f'{policy}_{rule}.json');stats=compare_evaluation(rec['evaluation'],ev_id['labels'])
                expected=snap.readout(rule)
                require(rec['aggregation']==expected,'G0 aggregation readout differs from raw update')
                require(rec['to_full']==compare_updates(increments[rule],ref,parent,p['scalings'],p['factor']),
                        'G0 full-update error differs from bank')
                rows.append({'round':r,'support':policy,'rule':rule,**{k:v for k,v in stats.items() if k!='predictions'},
                            'function_match_defined':expected['function_match_defined']})
        factual=snaps['pcla'].state('fixed')[0]
        if i+1==cp['completed_steps']:
            require(all(torch.equal(v,cp['adapter'][n]) for n,v in factual.items()),'G0 next parent differs')
        else:
            nxt=tensor_load(out/'rounds'/f'{r+1:03d}'/'parent.pt')['adapter']
            require(all(torch.equal(v,nxt[n]) for n,v in factual.items()),'G0 common parent path changed')
    done=cp['completed_steps']==2 and read(out/'status.json')['state']=='completed'
    require(done or not require_complete,'G0 incomplete')
    return {'kind':'fixed_state','complete':done,'seed':ident['seed'],'anchor':anchor,'rows':rows}


def audit(out,repo=None,require_complete=True):
    bind_repo(repo)
    out=Path(out);meta=read(out/'study.json')
    ev_id,part=evaluation_identity(out,repo)
    if meta['kind']=='aggregation':r=audit_aggregation(out,meta,ev_id,part,require_complete)
    elif meta['kind']=='lora_control':r=audit_control(out,meta,ev_id,part,require_complete,repo)
    elif meta['kind']=='fixed_state':r=audit_fixed_state(out,meta,ev_id,part,require_complete,repo)
    else:raise ValueError('unknown result kind')
    r.update({'audited':True,'audit_scope':'stored evidence/identities/metrics/algebra; no GPU forward rerun',
              'identity_hash':meta['identity_hash'],'evaluation_ids_sha256':fingerprint(ev_id['source_ids']),
              'partition_sha256':ev_id['partition_sha256'],'labels_sha256':fingerprint(ev_id['labels']),
              'checkpoint_sha256':sha256(out/('run/checkpoint.pt' if meta['kind']=='aggregation' else 'checkpoint.pt')),
              'audit_sources':source_files()})
    return r


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--repo')
    p.add_argument('--out',required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    atomic_json(a.out,audit(a.run,a.repo,not a.allow_partial))

if __name__=='__main__':main()
