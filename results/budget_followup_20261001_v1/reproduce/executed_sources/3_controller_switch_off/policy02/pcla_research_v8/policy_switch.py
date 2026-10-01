"""Validate a real accepted parent checkpoint for a fixed-horizon intervention."""
import hashlib,json
from pathlib import Path
import torch
from pcla_research_v8.server_control import _hash_state as tensor_digest

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()

def validate_policy(policy,job,server_controller):
 if (policy.get('parent_round')!=60 or policy.get('last_round')!=80 or job.get('rank')!=8
     or job.get('method')!='full_both' or job.get('representation')!='N'
     or job.get('recipe_id') not in {'U014','U018'} or policy.get('branch_id')!=f"OFF_{job['recipe_id']}_s{job['seed']}_r60_to80"
     or server_controller is not None):
  raise ValueError('policy-switch fixed scientific identity differs')
 if policy.get('schema')!='pcla-v8-policy-switch-1' or policy.get('alpha')!=1 or policy.get('beta')!=1:
  raise ValueError('policy must be ordinary complete update')
 if policy.get('first_round_technical_only') not in (None,61):raise ValueError('technical horizon must be first round61')

def load_parent(policy):
 path=Path(policy['parent_checkpoint_path'])
 if sha(path)!=policy['parent_checkpoint_file_sha256']:raise ValueError('parent checkpoint bytes changed')
 ck=torch.load(path,map_location='cpu',weights_only=True)
 required={'format','identity','round','adapter','metrics','cache','torch_rng','python_rng','cuda_rng','ema_effective','budget_stop_round','budget_adapter','half_adapter','half_round','half_readout','deployed_bytes','diagnostic_bytes'}
 if not required<=set(ck):raise ValueError('parent recovery state incomplete')
 if ck['format']!='pcla-v7-runtime-1' or ck['round']!=60 or ck['identity']!=policy['parent_checkpoint_identity']:raise ValueError('parent boundary or original identity differs')
 if tensor_digest(ck['adapter'])!=policy['parent_state_sha256']:raise ValueError('parent model state differs')
 if len(ck['metrics'])!=60 or [r['round'] for r in ck['metrics']]!=list(range(1,61)) or ck['budget_stop_round'] is not None:raise ValueError('parent prefix differs')
 if not {'clients','events'}<=set(ck['cache']) or len(ck['cache']['clients'])!=50:raise ValueError('parent cache incomplete')
 if not ck['cuda_rng'] or not ck['torch_rng'].numel() or any(not t.numel() for t in ck['cuda_rng']):raise ValueError('parent RNG incomplete')
 return ck
