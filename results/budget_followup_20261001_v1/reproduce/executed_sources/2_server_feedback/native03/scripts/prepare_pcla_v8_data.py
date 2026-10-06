import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcla_research_v8.data import extend_manifest,metadata_view
from pcla_repair.runtime import atomic_json
p=argparse.ArgumentParser();p.add_argument('--old-root',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
old=json.loads((a.old_root/'configs/core02/D066_s401.json').read_text())
manifest=json.loads(Path(old['manifest']).read_text());meta=json.loads(Path(old['token_cache']).with_suffix('.json').read_text())
seeds=[503,509,521,541,601,607,613,617,619,631,641,643,647,653,659,661,9003,9004]
new=extend_manifest(manifest,old['train_file'],seeds)
(a.root/'data').mkdir(parents=True,exist_ok=True)
atomic_json(a.root/'data/train_roles.json',new)
atomic_json(a.root/'data/train_tokens.json',metadata_view(meta,new))
old.update(manifest=str(a.root/'data/train_roles.json'),token_cache_metadata=str(a.root/'data/train_tokens.json'))
atomic_json(a.root/'data/loader_template.json',old)
print(json.dumps({'new_seed_count':len(seeds),'original_seeds':len(manifest['partitions']),'roles_unchanged':new['roles']==manifest['roles'],'split_sha256':new['split_sha256'],'token_cache_reused':old['token_cache']}))
