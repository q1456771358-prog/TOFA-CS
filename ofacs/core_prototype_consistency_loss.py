"""V574 lightweight source-only core prototype consistency regularizer."""
from __future__ import annotations
import torch
import torch.nn.functional as F

def compute_core_prototype_consistency_loss(embedding, query_sets, gt_sets, margin=0.05, topk=32, hard_neg_topk=32):
    if embedding is None or not query_sets:
        return embedding.new_tensor(0.0), {"pos_term":0.0,"neg_term":0.0,"count":0}
    losses=[]; pterm=[]; nterm=[]
    for q,gt in zip(query_sets,gt_sets):
        if not gt: continue
        ids=torch.tensor(list(gt),device=embedding.device,dtype=torch.long)
        core=embedding.index_select(0,ids)
        proto=core.mean(0,keepdim=True)
        sim=F.cosine_similarity(embedding,proto,dim=-1)
        pos=sim.index_select(0,ids)
        mask=torch.ones(sim.numel(),device=sim.device,dtype=torch.bool); mask[ids]=False
        neg=torch.topk(sim[mask],k=min(hard_neg_topk,int(mask.sum())),largest=True).values
        cp=torch.topk(pos,k=min(topk,pos.numel()),largest=True).values
        losses.append(F.relu(margin-cp[:,None]+neg[None,:]).mean())
        pterm.append(float(cp.mean())); nterm.append(float(neg.mean()))
    if not losses:
        return embedding.new_tensor(0.0), {"pos_term":0.0,"neg_term":0.0,"count":0}
    return torch.stack(losses).mean(), {"pos_term":sum(pterm)/len(pterm),"neg_term":sum(nterm)/len(nterm),"count":len(losses)}
