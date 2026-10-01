"""Offline synthetic integration fixture. Never use these outputs as evidence."""
from types import SimpleNamespace
import torch
from torch import nn
from torch.utils.data import Dataset
from .core import BudgetLoRALinear

class TinyDataset(Dataset):
    def __init__(self, seed: int, n: int = 24):
        rng=torch.Generator().manual_seed(seed)
        self.x=torch.randn(n,6,generator=rng)
        self.y=(self.x[:,0]+self.x[:,1]>0).long()
    def __len__(self): return len(self.y)
    def __getitem__(self,index): return {"x":self.x[index],"labels":self.y[index]}

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks=nn.ModuleList([nn.ModuleDict({
            "query":BudgetLoRALinear(nn.Linear(6,6),4,4.,.1),
            "value":BudgetLoRALinear(nn.Linear(6,6),4,4.,.1)}) for _ in range(4)])
        self.classifier=nn.Linear(6,2)
    def forward(self,x,labels):
        for block in self.blocks:
            x=torch.tanh(.5*block["query"](x)+.5*block["value"](x))+0.1*x
        logits=self.classifier(x)
        return SimpleNamespace(logits=logits,loss=nn.functional.cross_entropy(logits,labels))

def fixture(seed=123):
    torch.manual_seed(seed)
    model=TinyModel()
    layout={f"blocks.{i}.{p}":i for i in range(4) for p in ("query","value")}
    clients={i:TinyDataset(100+i) for i in range(4)}
    capacities={0:2,1:2,2:4,3:4}
    return model,layout,clients,capacities,TinyDataset(200,32)

def tiny_config():
    return dict(rounds=4, selected_clients=3, credits_per_layer=2, learning_rate=.01,
                betas=[.9,.999], adam_eps=1e-8, weight_decay=.01, local_epochs=1,
                batch_size=8, eval_batch_size=32, probe_batch_size=8, probe_batches=1,
                refresh_rounds=3, ema=.8,beta=.5,gamma=.05,score="normalized_projection",
                forward_mode="active_frozen",aggregation="fixed",train_head=True,
                clip_grad_norm=None,probe_rounds=[2,4])
