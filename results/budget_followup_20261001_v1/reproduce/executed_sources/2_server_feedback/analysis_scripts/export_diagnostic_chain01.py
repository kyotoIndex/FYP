"""Read-only diagnostic chain export from all 120 independently accepted objects."""
import argparse
import json
import math
from pathlib import Path
import export_validation_base01 as base

SOURCE_SHA = 'b4f1f1f117213e87b435417cc83fca948f26c37f70d6eb1c1321f7a5b0c1655b'
INDEX_SHA = '8edf6890bb50e4f4428f58ed54d8723368cc26328d994526b62e0eed77300ca2'
AUDITOR_SHA = 'bbdcfb5f4ffb65a9ed21260a2d468f88907aaea322b92274d6f24ce702eee4eb'
GATE_SHA = '2aee957f81bd41011e5f94f8f57b8afbbbd045a3206a0e0d27daf657f16b4564'
TECH_SHA = '480948197181dc640c00c07a6c8e6bc525df61e940512698a9c6481d5939f4d5'
SEEDS = [101,109,127,149,163,181,199,223,241,263,281,307]
RULES = ['ordinary_mean','scalar4','scalar17','block16','separable']


def close(a, b, label):
    base.require(math.isfinite(a) and math.isfinite(b) and
                 abs(a-b) <= 1e-9, label)


def expected_jobs():
    return ({f'BANK_{p}_r{n}_{f}_s{s}' for p in ['D069','D072']
             for n in [10,60] for f in ['a','b'] for s in SEEDS} |
            {f'R8_TRANSFER_r{n}_s{s}' for n in [10,60] for s in SEEDS})


def root_acceptances(root):
    accepted = {}; evidence = {}
    manifest = root/'sources/diagnostic_preflight02/SOURCE_MANIFEST.json'
    base.require(base.digest(manifest) == SOURCE_SHA, 'frozen diagnostic source manifest')
    index = root/'checks/diagnostic_intake01/INPUT_INDEX.json'
    base.require(base.digest(index) == INDEX_SHA, 'diagnostic input index')
    evidence.update({str(manifest):SOURCE_SHA, str(index):INDEX_SHA})
    for relative, expected_sha in base.read(manifest)['files'].items():
        path = manifest.parent/relative
        base.require(base.digest(path) == expected_sha, 'frozen diagnostic source file '+relative)
        evidence[str(path)] = expected_sha
    for ep in sorted((root/'checks/diagnostic_auditor02').glob('ROOT_ACCEPTANCE_EXECUTE_*.json')):
        e = base.read(ep); pp = ep.with_name(ep.name.replace('EXECUTE','PLAN'))
        base.require(e['accepted'] and base.digest(pp) == e['plan_sha256'], 'root diagnostic plan')
        p = base.read(pp); base.require(p['accepted'], 'root diagnostic review')
        evidence.update({str(ep):base.digest(ep), str(pp):base.digest(pp)})
        for jid in e['completed']:
            base.require(jid not in accepted, 'duplicate root diagnostic acceptance')
            item = next(x for x in p['items'] if x['job_id'] == jid)
            base.require(item['auditor_sha256'] == AUDITOR_SHA, 'diagnostic auditor identity')
            accepted[jid] = item
    base.require(set(accepted) == expected_jobs(), 'complete root accepted diagnostic matrix')
    return accepted, evidence


def project_feedback(identity, support, f):
    base.require(f['selection_role'] == 'public_training_512' and
                 f['independent_readout_role'] == 'cohort_readout_2048' and
                 f['selection_completed_before_independent_readout'] is True,
                 'public selection/readout ordering')
    base.require(set(f['rules']) == set(RULES), 'five diagnostic rules')
    corners = f['four_corners']; base.require(set(corners) == {'00','10','01','11'}, 'corner matrix')
    parent = corners['00']['independent_NLL']
    base.require(parent > 0 and math.isfinite(parent), 'parent loss denominator')
    ident = {**identity, 'support':support}
    rules = []
    for rule in RULES:
        row = f['rules'][rule]; payload = f['independent_readout_by_state_hash'][row['selected_state_hash']]
        close(payload['loss'], row['independent_NLL'], 'selected original readout')
        close(parent-row['independent_NLL'], row['independent_gain_vs_parent'], 'gain vs parent')
        base.require(payload['samples'] == 2048, 'independent readout denominator')
        rules.append({**ident,'rule':rule,'parent_independent_NLL':parent,
            **{k:row[k] for k in ['alpha','beta','selected_state_hash','public_NLL',
                'independent_NLL','public_NLL_source','public_gain_vs_parent',
                'independent_gain_vs_parent','public_independent_gain_direction_agrees',
                'standalone_measured_server_wall_seconds']},
            'independent_gain_fraction_parent': row['independent_gain_vs_parent']/parent,
            'raw_accuracy':payload['accuracy'],'samples':payload['samples'],
            'independent_NLL_benefit_vs_ordinary_mean':
                f['rules']['ordinary_mean']['independent_NLL']-row['independent_NLL'],
            'independent_NLL_benefit_vs_scalar17':
                f['rules']['scalar17']['independent_NLL']-row['independent_NLL']})
    chain = f['transfer_chain']; children = chain['clients']
    base.require(len(children) == 10 and len({c['client_id'] for c in children}) == 10,
                 'ten unique actual clients')
    clients = []
    for c in children:
        close(c['sample_weight'], .1, 'single sample weighting')
        close(c['local_parent']['loss']-c['local_child']['loss'], c['local_NLL_gain'], 'local gain')
        close(chain['global_parent']['loss']-c['individual_global']['loss'],
              c['individual_global_NLL_gain'], 'individual global gain')
        close(.1*c['individual_global_NLL_gain'], c['weighted_individual_global_NLL_gain'],
              'weighted exactly once')
        clients.append({**ident, 'client_id':c['client_id'], 'sample_weight':.1,
            'child_state_hash':c['child_state_hash'],
            'local_parent_NLL':c['local_parent']['loss'], 'local_child_NLL':c['local_child']['loss'],
            'local_samples':c['local_parent']['samples'],
            'local_NLL_gain':c['local_NLL_gain'],
            'individual_global_NLL':c['individual_global']['loss'],
            'individual_global_NLL_gain':c['individual_global_NLL_gain'],
            'weighted_individual_global_NLL_gain':c['weighted_individual_global_NLL_gain']})
    weighted = math.fsum(c['weighted_individual_global_NLL_gain'] for c in children)
    close(weighted, chain['weighted_individual_global_NLL_gain'], 'sum of weighted gains')
    close(parent-chain['aggregate']['loss'], chain['aggregate_global_NLL_gain'], 'aggregate gain')
    close(chain['aggregate']['loss'], f['rules']['ordinary_mean']['independent_NLL'], 'mean aggregate')
    corner_rows = [{**ident,'corner':k,**corners[k]} for k in ['00','10','01','11']]
    interactions = {**ident, 'parent_independent_NLL':parent,
        'weighted_individual_global_NLL_gain':weighted,
        'aggregate_global_NLL_gain':chain['aggregate_global_NLL_gain'],
        'aggregate_minus_sum_individual':chain['aggregate_global_NLL_gain']-weighted,
        'public_interaction_I':f['interaction_I']['public'],
        'independent_interaction_I':f['interaction_I']['independent'],
        'posthoc_best_independent_NLL':f['posthoc_readout_grid']['best_independent_NLL'],
        'posthoc_role':f['posthoc_readout_grid']['role']}
    for role in ['public','independent']:
        close(corners['10'][role+'_NLL']+corners['01'][role+'_NLL']-
              corners['11'][role+'_NLL']-corners['00'][role+'_NLL'],
              f['interaction_I'][role], 'four-corner interaction')
    return rules, clients, corner_rows, interactions


def load_job(qp, root_items):
    q = base.read(qp); jid = q['job_id']; item = root_items[jid]
    cp, rp, ap = map(Path, [q['config'],q['result_path'],q['audit_receipt_path']])
    c, v, a = map(base.read,[cp,rp,ap])
    base.require(q['state'] == 'completed' and q['return_code'] == 0 and
                 a['status'] == 'accepted', 'completed audited diagnostic')
    base.require(base.digest(cp) == q['config_sha256'] == v['config_sha256'] == a['config_sha256'], 'diagnostic config hash')
    base.require(base.digest(rp) == q['result_sha256'] == a['result_sha256'], 'diagnostic result hash')
    base.require(base.digest(ap) == q['audit_receipt_sha256'] == item['receipt_sha256']
                 and str(ap) == item['receipt_path'], 'root diagnostic receipt binding')
    base.require(jid == a['job_id'] == v['job_id'] == c['job']['job_id'] and
                 c['job'] == v['job'] and v['source_manifest_sha256'] == a['source_manifest_sha256'] == SOURCE_SHA
                 and v['input_index_sha256'] == a['input_index_sha256'] == INDEX_SHA,
                 'diagnostic identity/source/index')
    ph = v['job']['phase']; supports = ['local','cyclic'] if ph == 'BANK_READOUT' else ['local']
    base.require(set(v['reports']) == set(supports), 'support coverage')
    calls, steps = (0,0) if ph == 'BANK_READOUT' else (10,160)
    base.require(v['collection']['local_calls'] == a['new_local_calls'] == calls and
                 v['collection']['optimizer_steps'] == a['new_optimizer_steps'] == steps,
                 'training reuse/update denominator')
    identity = {**v['job'],'gpu_uuid':q['claim']['gpu_uuid'],
                'source_object_id':q.get('source_object_id')}
    evidence = {str(p):base.digest(p) for p in [qp,cp,rp,ap]}
    for path_key, hash_key in [('root_gate_path','root_gate_sha256'),
                              ('independent_technical_receipt_path','independent_technical_receipt_sha256')]:
        path=Path(q[path_key]); expected=GATE_SHA if path_key=='root_gate_path' else TECH_SHA
        base.require(base.digest(path) == q[hash_key] == expected, 'diagnostic gate binding'); evidence[str(path)]=q[hash_key]
    output=[]; support_records={}
    for support in supports:
        fp=rp.parent/f'feedback_{support}.json'; f=base.read(fp)
        base.require(base.digest(fp) == v['evidence'][fp.name] and f == v['reports'][support], 'feedback original hash/content')
        evidence[str(fp)]=base.digest(fp)
        output.append(project_feedback(identity,support,f))
        support_records[support]={'counters_original':f['counters'], 'timers_original':f['timers'],
            'posthoc_readout_grid_original':f['posthoc_readout_grid'],
            'rule_controllers_original':{k:f['rules'][k]['controller'] for k in RULES}}
    metadata = {'identity':identity,'collection_original':v['collection'],
                'elapsed_seconds':v['elapsed_seconds'],'input_receipt_original':v['input_receipt'],
                'old_reference_provenance':{k:z for k,z in v.get('old_reference',{}).items() if k != 'raw'},
                'evidence_sha256_at_export':evidence,
                'supports':supports,'support_records':support_records,
                'audit_original':a,'queue_original':q}
    return output, metadata


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(); items,evidence=root_acceptances(args.root)
    paths={p.stem:p for p in (args.root/'queues/diagnostic01/jobs').glob('*.json')}
    base.require(set(paths)==expected_jobs() and not args.out.exists(), 'new complete diagnostic export')
    records=[load_job(paths[j],items) for j in sorted(paths)]
    # No PT reads or model calls. All original bindings validated before formal output creation.
    tables=[[],[],[],[]]; metadata=[]
    for outputs,m in records:
        metadata.append(m);evidence.update(m['evidence_sha256_at_export'])
        for output in outputs:
            for i,x in enumerate(output): tables[i].extend(x if isinstance(x,list) else [x])
    base.require(len(metadata)==120 and len(tables[0])==1080 and len(tables[1])==2160,
                 'diagnostic object/support/client denominator')
    args.out.mkdir(parents=True)
    names=['rules.csv','clients.csv','four_corners.csv','transfer_interactions.csv']
    # Phase-specific optional identity fields use a common union of columns.
    for name,rows in zip(names,tables):
        keys=list(dict.fromkeys(k for row in rows for k in row))
        base.write_csv(args.out/name,[{k:row.get(k) for k in keys} for row in rows])
    with (args.out/'metadata_originals.jsonl').open('x') as f:
        for m in metadata:f.write(json.dumps(m,allow_nan=False)+'\n')
    (args.out/'INPUT_HASHES.json').write_text(json.dumps(evidence,indent=2))
    manifest={'schema':'pcla-v8-diagnostic-chain-export-1','objects':120,'BANK_objects':96,'R8_objects':24,
        'supports':216,'rule_rows':1080,'client_rows':2160,'new_training_calls':240,'new_optimizer_steps':3840,
        'statistical_unit':'seed; supports/parents/clients not independent replicates',
        'seed_normalization':'object ratio then within-seed equal mean; seed sum ratios reported separately',
        'posthoc_grid':'diagnostic only; not executable selection',
        'cost_scope':'standalone server times in rule rows; separate readout/collection/operating ledger required',
        'source_manifest_sha256':SOURCE_SHA,'input_index_sha256':INDEX_SHA,
        'export_code_sha256':base.digest(__file__),'files':{x.name:base.digest(x) for x in args.out.iterdir()}}
    (args.out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'objects':120,'support_rows':216,'out':str(args.out)}))


if __name__=='__main__':main()
