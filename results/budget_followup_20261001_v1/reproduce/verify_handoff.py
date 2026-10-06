#!/usr/bin/env python3
"""Independent integrity checks for the reader tables and actual-config handoff."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics as st
import urllib.parse

ROOT=Path(__file__).resolve().parents[1]

def rows(p):
    with open(ROOT/p,encoding='utf-8') as f:return list(csv.DictReader(f))

def check(condition,message):
    if not condition:raise AssertionError(message)

def main():
    expected={'component_development.csv':117,'server_development.csv':32,'component_validation_seeds.csv':192,'server_validation_seeds.csv':108,'component_methods.csv':16,'server_methods.csv':9,'key_paired_comparisons.csv':14,'switch_off_seed_pairs.csv':96,'coordinate_diagnostics.csv':60}
    for file,n in expected.items():
        rs=rows('reader/'+file);check(len(rs)==n,f'{file}: row count')
        check(not re.search(r'\b(?:D\d{3}|U\d{3}|COST1|BANK_READOUT|R8_TRANSFER)\b',(ROOT/'reader'/file).read_text()),f'{file}: internal labels')
    for study,source,field in [('component','data/1_budget_selection/validation/v_endpoints.csv','b8_accuracy'),('server','data/2_server_feedback/validation/endpoints.csv','raw_B8_accuracy')]:
        original=rows(source);published=rows('reader/'+study+'_validation_seeds.csv')
        for a,b in zip(original,published):
            check(int(a['seed'])==int(b['seed']),f'{study}: seed ordering')
            check(abs(100*float(a[field])-float(b['accuracy_percent']))<1e-10,f'{study}: accuracy mapping')
        grouped={}
        for r in published:grouped.setdefault(r['method'],[]).append(r)
        check(all(len(v)==12 and len({r['seed'] for r in v})==12 for v in grouped.values()),f'{study}: unique seeds')
    index=rows('reproduce/actual_configs/INDEX.csv');check(len(index)==428,'config matrix')
    check(len({r['config_file'] for r in index})==428,'unique configuration paths')
    manifest=json.loads((ROOT/'reproduce/PUBLIC_MANIFEST.json').read_text());entries={e['published']:e for e in manifest['files']}
    check(len(entries)==len(manifest['files']),'unique manifest entries')
    for row in index:
        p=ROOT/row['config_file'];check(p.is_file(),'actual configuration exists')
        check(row['config_file'] in entries,'actual configuration listed')
        if row['study']=='组件选择':
            check(entries[row['config_file']]['source_sha256']==row['original_config_sha256'],'component config execution SHA')
        c=json.loads(p.read_text());check(int(c['job']['seed'])==int(row['seed']),'config seed')
    for phase,n in [('development',128),('validation',108)]:
        archive=ROOT/f'data/2_server_feedback/{phase}/accepted_originals.jsonl.gz'
        bindings=[r for r in index if r['study']=='服务器更新控制' and r['phase']==('开发' if phase=='development' else '验证')]
        bysha={r['original_config_sha256']:r for r in bindings};count=0
        with gzip.open(archive,'rt') as f:
            for line in f:
                d=json.loads(line);h=d.get('endpoint',{}).get('config_sha256') or d['result_original']['config_sha256']
                check(h in bysha,'config original-hash binding')
                c=json.loads((ROOT/bysha[h]['config_file']).read_text())
                check(c==d['config_original'],'exact embedded config extraction')
                count+=1
        check(count==len(bindings)==n,'executed config coverage')
    build=json.loads((ROOT/'reader/BUILD_MANIFEST.json').read_text())
    for section in ('inputs','outputs'):
        for name,h in build[section].items():check(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,'reader build hash')
    pairs=rows('reader/switch_off_seed_pairs.csv')
    for method in {r['method'] for r in pairs}:
        end=[r for r in pairs if r['method']==method and int(r['round'])==80]
        check(len(end)==12 and sum(float(r['accuracy_difference_pp'])>0 for r in end)==7,'switch off distribution')
        check(sum(float(r['NLL_difference'])<0 for r in end)==7,'switch off NLL distribution')
    docs=['README.md','METHODS.md','HANDOFF.md','SOURCES.md','reproduce/README.md']
    for file in docs:
        text=(ROOT/file).read_text();check(not re.search('我们|大家',text),f'{file}: singular authorship')
        # URL/path targets remain exact archive paths; visible prose uses scientific names.
        prose=re.sub(r'\]\([^)]*\)','](link)',text)
        check(not re.search(r'\b(?:D\d{3}|U\d{3}|core\d{2}|native\d{2}|V[78])\b',prose),f'{file}: visible internal code')
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if '://' in target or target.startswith('#'):continue
            path=urllib.parse.unquote(target.split('#')[0]);check(((ROOT/file).parent/path).exists(),f'{file}: broken link {target}')
    print('PASS: 428 unique executed configurations, 149 development recipes, 300 validation endpoints, 24 paired branches, reader hashes and handoff links')

if __name__=='__main__':main()
