"""Focused regressions for the newly added learning-rate and readout code."""
import copy,importlib.util,json,sys
from pathlib import Path
import pytest,torch
from pcla_repair.core import adapter_state,load_adapter,lora_modules,configure
from pcla_repair.runtime import local_train,run_experiment,load_checkpoint,evaluate
from pcla_repair.testing import fixture,tiny_config
from pcla_repair.optimization_dev import make_optimizer,classification_readout,constant_references

@pytest.mark.parametrize('factor',['a','b'])
def test_separate_rates_still_freeze_inactive_entries(factor):
 model,layout,clients,*_=fixture()
 for m in lora_modules(model).values():m.lora_b.data.fill_(.3)
 before=adapter_state(model);cfg=tiny_config();cfg.update(head_learning_rate=.0001,lora_learning_rate=.001)
 support=torch.zeros(4,4,dtype=torch.bool);support[0,0]=True;d={}
 masks,_,_=local_train(model,layout,clients[0],support,factor,17,cfg,torch.device('cpu'),diagnostic=d)
 after=adapter_state(model)
 for n,value in after.items():
  m=masks.get(n,torch.zeros_like(value,dtype=torch.bool))
  assert torch.equal(before[n][~m],value[~m])
 assert d['optimizer_steps']>0 and d['effective_head_learning_rate']==.0001


def test_logging_has_no_effect_on_training_values_or_rng():
 a,layout,clients,*_=fixture();b=copy.deepcopy(a);cfg=tiny_config()
 mask=torch.ones(4,4,dtype=torch.bool)
 local_train(a,layout,clients[0],mask,'b',71,cfg,torch.device('cpu'))
 rng=torch.get_rng_state().clone();d={}
 local_train(b,layout,clients[0],mask,'b',71,cfg,torch.device('cpu'),diagnostic=d)
 assert torch.equal(rng,torch.get_rng_state())
 for k,v in adapter_state(a).items():assert torch.equal(v,adapter_state(b)[k])
 assert d['training_examples_seen']==len(clients[0])*cfg['local_epochs']


def test_same_rate_factory_preserves_original_single_group_and_updates():
 a,layout,*_=fixture();b=copy.deepcopy(a);cfg=tiny_config();mask=torch.ones(4,4,dtype=torch.bool)
 configure(a,layout,mask,'b');configure(b,layout,mask,'b')
 pa=[p for p in a.parameters() if p.requires_grad];pb=[p for p in b.parameters() if p.requires_grad]
 original=torch.optim.AdamW(pa,lr=cfg['learning_rate'],betas=tuple(cfg['betas']),eps=cfg['adam_eps'],weight_decay=cfg['weight_decay'],foreach=False)
 new=make_optimizer(b,{**cfg,'head_learning_rate':cfg['learning_rate'],'lora_learning_rate':cfg['learning_rate']})
 assert len(new.param_groups)==1
 for _ in range(3):
  for i,(x,y) in enumerate(zip(pa,pb)):
   x.grad=torch.full_like(x,.02*(i+1));y.grad=x.grad.clone()
  original.step();new.step()
 for x,y in zip(pa,pb):assert torch.equal(x,y)


def test_parameter_groups_have_requested_rates():
 m,layout,*_=fixture();cfg=tiny_config();configure(m,layout,torch.ones(4,4,dtype=torch.bool),'a')
 opt=make_optimizer(m,{**cfg,'head_learning_rate':.0001,'lora_learning_rate':.001})
 head_ids={id(p) for p in m.classifier.parameters()}
 for group in opt.param_groups:
  for p in group['params']:assert group['lr']==(.0001 if id(p) in head_ids else .001)


def test_head_only_ignores_unused_lora_rate():
 a,layout,clients,*_=fixture();b=copy.deepcopy(a);mask=torch.zeros(4,4,dtype=torch.bool);cfg=tiny_config()
 for model,lora in [(a,.001),(b,.0001)]:
  local_train(model,layout,clients[0],mask,'none',99,{**cfg,'head_learning_rate':.001,'lora_learning_rate':lora},torch.device('cpu'))
 for n,t in adapter_state(a).items():assert torch.equal(t,adapter_state(b)[n])


def test_logits_readout_describes_constant_predictions():
 labels=torch.tensor([0,1,1,0]);logits=torch.tensor([[0.,3.]]*4)
 r=classification_readout(logits,labels)
 assert r['predicted_class_counts']==[0,4] and r['balanced_accuracy']==.5
 assert r['confusion']==[[0,2],[0,2]] and r['class_recall']==[0.,1.]
 assert r['nll_recomputed_float64']>1.5


def test_observation_does_not_change_evaluation_loss():
 m,layout,_,_,ev=fixture()
 x=evaluate(m,layout,ev,8,torch.device('cpu'))
 y=evaluate(m,layout,ev,8,torch.device('cpu'),record_logits=True)
 assert x['loss']==y['loss'] and x['predictions']==y['predictions']


def test_development_requires_tuning_or_explicit_synthetic(tmp_path):
 m,layout,caps,cap,ev=fixture();cfg={**tiny_config(),'record_optimization_diagnostics':True}
 with pytest.raises(ValueError,match='tuning'):
  run_experiment(m,layout,caps,cap,ev,cfg,7,'pcla',tmp_path/'bad',{'evaluation_split':'benchmark_dev'})


def test_new_readouts_preserve_trajectory_and_resume(tmp_path):
 cfg=tiny_config()
 def go(name,record,stop=None):
  m,layout,clients,caps,ev=fixture()
  return run_experiment(m,layout,clients,caps,ev,{**cfg,'record_optimization_diagnostics':record},7,'pcla',tmp_path/name,{'synthetic':True},stop_after=stop)
 go('plain',False);go('logged',True);go('resume',True,2);go('resume',True)
 a,b,c=[load_checkpoint(tmp_path/name/'checkpoint.pt') for name in ('plain','logged','resume')]
 for n in a['adapter']:assert torch.equal(a['adapter'][n],b['adapter'][n]) and torch.equal(b['adapter'][n],c['adapter'][n])
 for n in a['trace']:assert torch.equal(a['trace'][n],b['trace'][n]) and torch.equal(b['trace'][n],c['trace'][n])
 assert [r['loss'] for r in a['metrics']]==[r['loss'] for r in b['metrics']]==[r['loss'] for r in c['metrics']]
 assert (tmp_path/'logged/evaluation_labels.json').exists()
 assert (tmp_path/'logged/logits_round_004.json').exists()


def test_remix_preserves_exact_sources_and_counts():
 path=Path(__file__).resolve().parents[1]/'scripts/diagnose_pcla_common_failure.py'
 spec=importlib.util.spec_from_file_location('diag_common',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 _,_,clients,*_=fixture()
 selected=sorted(clients)
 refs=mod.remix_refs(clients,selected,123)
 union=[x for g in refs.values() for x in g]
 assert len(union)==len(set(union))==sum(len(clients[c]) for c in selected)
 assert refs==mod.remix_refs(clients,selected,123)
 neg=[sum(int(clients[c][i]['labels'])==0 for c,i in g) for g in refs.values()]
 assert max(neg)-min(neg)<=1

@pytest.mark.parametrize('lr',[0,-.1,float('nan'),float('inf')])
def test_invalid_rate_is_rejected(lr):
 m,layout,*_=fixture();configure(m,layout,torch.ones(4,4,dtype=torch.bool),'b')
 with pytest.raises(ValueError,match='positive'):
  make_optimizer(m,{**tiny_config(),'head_learning_rate':lr})
