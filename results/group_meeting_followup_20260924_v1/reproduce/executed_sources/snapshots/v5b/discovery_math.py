"""Reusable, tested numerical primitives; no claim of a complete Qwen backend."""
from __future__ import annotations
import math
import torch
from common import require
from lora_matrix import validate_pair,product_delta,stacked_exact


def projected_moment(a,batches):
    a=a.detach().cpu().double()
    require(a.ndim==2 and bool(torch.isfinite(a).all()),'invalid input factor')
    c=torch.zeros(a.shape[0],a.shape[0],dtype=torch.float64);count=0
    for x in batches:
        x=x.detach().cpu().double()
        require(x.ndim==2 and x.shape[1]==a.shape[1] and bool(torch.isfinite(x).all()),'invalid unpadded activation rows')
        z=x@a.T;c+=z.T@z;count+=len(z)
    require(count>0,'empty activation set')
    return c/count,count


def moment_checked(c,size):
    c=c.detach().cpu().double()
    require(c.shape==(size,size) and bool(torch.isfinite(c).all()),'invalid projected moment')
    c=(c+c.T)/2
    tol=64*torch.finfo(torch.float64).eps*max(1.,float(c.norm()))*size
    require(float(torch.linalg.eigvalsh(c).min())>=-tol,'moment is not positive semidefinite')
    return c


def compress(b,a,rank,moment=None):
    b,a=validate_pair(b,a)
    require(isinstance(rank,int) and 0<=rank<=min(b.shape[0],b.shape[1],a.shape[1]),'invalid compression rank')
    if rank==0:return b[:,:0].clone(),a[:0].clone()
    c=a@a.T if moment is None else moment_checked(moment,a.shape[0])
    if moment is not None:require(float(c.trace())>0,'zero functional energy: no informed compression selection')
    q,r=torch.linalg.qr(b,mode='reduced');h=r@c@r.T
    _,u=torch.linalg.eigh((h+h.T)/2);u=u[:,-rank:]
    return q@u,u.T@r@a


def group_reference(b,a,moments,rank,iterations=20,step=.5):
    """Bounded calibration heuristic; neither a new algorithm claim nor a global optimum."""
    b,a=validate_pair(b,a)
    require(0<rank<=min(b.shape[0],b.shape[1],a.shape[1]),'invalid rank')
    require(moments and isinstance(iterations,int) and iterations>0 and math.isfinite(step) and step>0,'invalid reference setting')
    q,r=torch.linalg.qr(b,mode='reduced');hs=[];energies=[];floors=[]
    for c in moments:
        c=moment_checked(c,a.shape[0]);h=r@c@r.T;h=(h+h.T)/2;e=float(h.trace())
        require(e>0,'group has zero functional energy')
        hs.append(h);energies.append(e)
        floors.append(max(0.,float(torch.linalg.eigvalsh(h)[:-rank].sum()))/e)
    w=torch.ones(len(hs),dtype=torch.float64)/len(hs);best=math.inf;best_u=None;history=[]
    for i in range(iterations):
        h=sum(x*y/e for x,y,e in zip(w,hs,energies));_,u=torch.linalg.eigh((h+h.T)/2);u=u[:,-rank:]
        loss=torch.tensor([max(0.,float(h.trace()-(u.T@h@u).trace()))/e-f for h,e,f in zip(hs,energies,floors)],dtype=torch.float64)
        value=float(loss.max())
        if value<best:best,best_u=value,u.clone()
        history.append({'iteration':i,'weights':w.tolist(),'group_excess':loss.tolist(),'max_excess':value,'best_so_far':best})
        w*=torch.exp(step*(loss-loss.max()));w/=w.sum()
    return q@best_u,best_u.T@r@a,history
