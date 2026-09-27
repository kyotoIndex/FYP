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


def audit_ffa_bank(bank,record,part,reference_a,probe):
    require(bank['factor']=='b' and [c['client'] for c in bank['clients']]==record['selected_clients'],'FFA bank treatment differs')
    require(all(c['weight']==len(part['clients'][str(c['client'])]) for c in bank['clients']),'FFA client weight differs')
    for state in [bank['parent_adapter']]+[c['adapter'] for c in bank['clients']]:
        require(all(torch.equal(state[n],v) for n,v in reference_a.items()),'FFA bank A differs from shared initial A')
    from pcla_repair.core import tensor_digest
    base=bank['parent_adapter'];total=sum(c['weight'] for c in bank['clients'])
    require(tensor_digest(base)==record['initial_round_adapter_sha256'],'FFA bank parent differs from round start')
    mean={n:(v.double()+sum(c['weight']*(c['adapter'][n].double()-v.double()) for c in bank['clients'])/total).to(v.dtype)
          for n,v in base.items()}
    require(tensor_digest(mean)==record['post_round_adapter_sha256'],'FFA B/head aggregate differs from fixed-denominator FedAvg')
    if probe is not None:require(all(torch.equal(mean[n],v) for n,v in probe.items()),'FFA probe differs from aggregate')


def audit_control(out,meta,ev_id,part,require_complete,repo=None):
    out=Path(out);cp=tensor_load(out/'checkpoint.pt');identity=cp['identity'];cfg=identity['config'];seed=identity['seed']
    require(fingerprint(identity)==cp['identity_hash']==meta['identity_hash'],'control identity differs')
    require(identity==meta['identity'] and identity['source_files']==source_files(),'control source differs')
    verify_files(out,[row['evidence'] for row in cp['metrics']])
    require([x['round'] for x in cp['metrics']]==list(range(1,cp['round']+1)),'control rounds incomplete')
    mode=identity['mode'];rows=[];labels=client_labels(part,ev_id,repo) if mode=='central_iid_both' else None
    reference_a=None;previous=identity['initial_adapter_sha256']
    if mode=='fed_ffa_b':
        from pcla_repair.core import tensor_digest as digest
        from run_lora_controls import FFA_FIDELITY
        first=tensor_load(out/'rounds/001/local_bank.pt')['parent_adapter']
        require(digest(first)==identity['initial_adapter_sha256'],'FFA initial adapter differs')
        reference_a={n:v for n,v in first.items() if n.endswith('.lora_a')}
        require(all(torch.equal(cp['adapter'][n],v) for n,v in reference_a.items()),'FFA final A differs from shared initial A')
    for row in cp['metrics']:
        r=row['round'];record=read(out/'rounds'/f'{r:03d}'/'record.json')
        participants=selected(seed,r,part,cfg)
        require(record['mode']==mode and record['selected_clients']==participants,'control treatment/participants differ')
        if mode=='central_iid_both':audit_iid_round(record,seed,r,part,cfg,participants,labels)
        else:require(record['sample_multiset_sha256']==fingerprint([s for c in participants for s in part['clients'][str(c)]]),
                     'control sample exposure differs')
        if mode=='fed_ffa_b':
            require(record['factor']=='b' and all(c['factor']=='b' for c in record['client_details']),'FFA trained factor differs')
            require(record['lora_a_sha256']==digest(reference_a) and record['fidelity']==FFA_FIDELITY,'FFA record differs')
            require(record['initial_round_adapter_sha256']==previous,'FFA round chain broken')
            previous=record['post_round_adapter_sha256']
            require(all(x['residual_norm']<1e-4*max(1.,x['target_norm']) for x in record['algebra']),'FFA factor mean is not exact')
        expected_steps=sum(math.ceil(len(part['clients'][str(c)])/cfg['batch_size']) for c in participants)*cfg['local_epochs']
        expected_samples=sum(len(part['clients'][str(c)]) for c in participants)*cfg['local_epochs']
        require(record['steps']==expected_steps and record['examples']==expected_samples,'control budget differs')
        stats=compare_evaluation(record['evaluation'],ev_id['labels'])
        require(row['accuracy']==stats['accuracy'] and row['gpu_nll']==stats['gpu_nll'],'control row differs')
        if 'same_update_alternative' in record:
            compare_evaluation(record['same_update_alternative']['evaluation'],ev_id['labels'])
        bank=out/'rounds'/f'{r:03d}'/'local_bank.pt'
        if bank.exists():audit_joint_bank(tensor_load(bank),identity['mode'],record)
        if mode=='fed_ffa_b' and bank.exists():
            probe=out/f'probe_round_{r:03d}.pt'
            audit_ffa_bank(tensor_load(bank),record,part,reference_a,tensor_load(probe)['adapter'] if probe.exists() else None)
        rows.append({'round':r,**{k:v for k,v in stats.items() if k!='predictions'},
                     'train_and_server_seconds':record['training_and_server_seconds'],
                     'upload_bytes':record['upload_bytes'],'download_bytes':record['download_bytes'],
                     **{k:record[k] for k in ('iid_positive_fraction','cohort_positive_fraction','download_bytes_b_head_only') if k in record}})
    from pcla_repair.core import tensor_digest
    if mode=='fed_ffa_b':require(previous==tensor_digest(cp['adapter']),'FFA final adapter differs from last round')
    if identity['mode']!='fed_both_exact':
        require(tensor_digest(cp['projections'])==identity['initial_projections_sha256'],'frozen base changed')
    last=out/f"probe_round_{cp['round']:03d}.pt"
    if last.exists():
        snap=tensor_load(last)
        require(snap['identity_hash']==cp['identity_hash'],'last probe identity differs')
        require(tensor_digest(cp['adapter'])==tensor_digest(snap['adapter']) and
                tensor_digest(cp['projections'])==tensor_digest(snap['projections']),'last checkpoint/probe differ')
    done=read(out/'status.json')['state']=='completed' and cp['round']==cfg['rounds']
    require(done or not require_complete,'control horizon incomplete')
    return {'kind':'lora_control','complete':done,'seed':seed,'mode':identity['mode'],'rows':rows}


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
