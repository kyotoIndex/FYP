"""Independent small-file checks and last-five-round pairs for completed V5d."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

SEEDS=(7,13,21,42,87)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def audit(root, name, batch, source_files):
    path=root/name/'audit.json'
    x=read(path)
    assert x['complete'] and len(x['rows'])==30 and x['audit_sources']==source_files
    assert [v['round'] for v in x['rows']]==list(range(1,31))
    return {'last5_accuracy':sum(row['accuracy'] for row in x['rows'][-5:])/5,
            'audit_sha256':sha(path),'archive_sha256':batch['archive_sha256']}


def quotas(value):
    return [sum(layer) for layer in value]


def central_check(donor, replay):
    for r in range(1,31):
        a=read(donor/'run'/'rounds'/f'{r:03d}'/'record.json')
        b=read(replay/'run'/'rounds'/f'{r:03d}'/'record.json')
        assert quotas(a['support'])==quotas(b['support'])
        assert a['selected_clients']==b['selected_clients']
        assert a['sample_multiset_sha256']==b['sample_multiset_sha256']
        assert a['pooled_order_sha256']==b['pooled_order_sha256']


def svd_check(donor, replay):
    for r in range(1,31):
        a=read(donor/'run'/'run'/f'clients_round_{r:03d}.json')
        b=read(replay/'run'/'run'/f'clients_round_{r:03d}.json')
        assert len(a)==len(b)==10
        for x,y in zip(a,b):
            assert x['client']==y['client'] and x['factor']==y['factor']
            assert quotas(x['support'])==quotas(y['support'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();root=Path(a.root);out=Path(a.out)
    results={}
    for variant in ('central','svd'):
        queue=root/'queue_v5d'/variant;materialized=root/'materialized_v5d'/variant
        frozen=read(queue/'FROZEN.json')
        inventory=read(materialized/'inventory.json')
        assert len(inventory)==5 and all(x['state']=='completed' for x in inventory)
        assert len(frozen['jobs'])==5
        rows=[]
        for seed in SEEDS:
            job=(f'C_central_replay_random_rank_r30_s{seed}' if variant=='central'
                 else f'V5_replay_svd_s{seed}')
            donor_job=(f'C_central_pcla_r30_s{seed}' if variant=='central'
                       else f'V4_pcla_svd_s{seed}')
            status=read(queue/'status'/f'{job}.json')
            assert status['state']=='completed' and sha(status['receipt'])==status['receipt_sha256']
            receipt=read(status['receipt'])
            assert receipt['state']=='completed' and receipt['job']['id']==job
            archive_sha=receipt['archive']['sha256']
            assert read(materialized/job/'EXTRACTION_RECEIPT.json')['archive_sha256']==archive_sha
            droot=root/('materialized_v5a' if variant=='central' else 'materialized_v5b')/donor_job
            rroot=materialized/job
            donor_audit=read(droot/'audit.json')
            assert donor_audit['complete'] and donor_audit['seed']==seed and len(donor_audit['rows'])==30
            counterpart=audit(materialized,job,{'archive_sha256':archive_sha},frozen['source_files'])
            assert read(rroot/'audit.json')['seed']==seed
            if variant=='central':central_check(droot,rroot)
            else:svd_check(droot,rroot)
            donor_mean=sum(x['accuracy'] for x in donor_audit['rows'][-5:])/5
            rows.append({'seed':seed,'donor_job':donor_job,'new_job':job,
                         'donor_last5_accuracy':donor_mean,
                         'new_last5_accuracy':counterpart['last5_accuracy'],
                         'donor_minus_new_percentage_points':100*(donor_mean-counterpart['last5_accuracy']),
                         'donor_audit_sha256':sha(droot/'audit.json'),
                         'new_audit_sha256':counterpart['audit_sha256'],
                         'new_archive_sha256':archive_sha})
        results[variant]={'completed':5,'rows':rows,'donor_minus_new_mean_percentage_points':
                          sum(x['donor_minus_new_percentage_points'] for x in rows)/5,
                          'checks':'30 rounds x 5 seeds: same layer quotas and training exposure'}
    assert not out.exists(),'new summary file required'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({v:{'completed':x['completed'],
                         'donor_minus_new_mean_percentage_points':x['donor_minus_new_mean_percentage_points']}
                      for v,x in results.items()},indent=2))


if __name__=='__main__':main()
