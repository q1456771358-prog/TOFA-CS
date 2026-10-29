"""V574 source-only state conditioned boundary loss.

No target graph labels are consumed.  The query state is estimated from the
source query prediction against source ground truth only.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class StateBoundaryLossReport:
    state_counts: dict
    pos_term: float
    neg_term: float
    size_term: float
    def to_dict(self):
        return self.__dict__.copy()

def compute_state_conditioned_boundary_loss(embedding, query_sets, gt_sets, margin=0.05, topk_pos=64, topk_neg=64, under_recall_threshold=0.25, over_precision_threshold=0.70):
    if embedding is None or not query_sets:
        return embedding.new_tensor(0.0), StateBoundaryLossReport({},0.0,0.0,0.0)
    losses=[]; counts={"under":0,"moderate":0,"boundary":0,"stable":0}
    pos_vals=[]; neg_vals=[]
    for q, gt in zip(query_sets, gt_sets):
        if len(gt)==0: continue
        qids=torch.tensor(list(q),device=embedding.device,dtype=torch.long)
        gids=torch.tensor(list(gt),device=embedding.device,dtype=torch.long)
        qemb=embedding.index_select(0,qids).mean(0,keepdim=True)
        scores=F.cosine_similarity(embedding,qemb,dim=-1)
        pos=scores.index_select(0,gids)
        neg_mask=torch.ones(scores.numel(),device=scores.device,dtype=torch.bool)
        neg_mask[gids]=False
        neg=scores[neg_mask]
        miss=float((pos<pos.mean()).float().mean().item())
        fp=float((neg>pos.mean()).float().mean().item())
        if miss>under_recall_threshold: state="under"
        elif fp>over_precision_threshold: state="boundary"
        elif abs(miss-fp)<0.15: state="stable"
        else: state="moderate"
        counts[state]+=1
        if state=="under": wp,wn=1.5,0.5
        elif state=="moderate": wp,wn=1.0,0.8
        elif state=="boundary": wp,wn=0.5,1.5
        else: wp,wn=0.3,0.3
        p=torch.topk(pos,k=min(topk_pos,pos.numel()),largest=False).values
        n=torch.topk(neg,k=min(topk_neg,neg.numel()),largest=True).values
        losses.append(wp*F.relu(margin-p[:,None]+n[None,:]).mean()+wn*F.relu(margin-p.mean()+n.mean()))
        pos_vals.append(float(p.mean().detach())); neg_vals.append(float(n.mean().detach()))
    if not losses:
        return embedding.new_tensor(0.0), StateBoundaryLossReport(counts,0.0,0.0,0.0)
    loss=torch.stack(losses).mean()
    return loss, StateBoundaryLossReport(counts,float(sum(pos_vals)/max(1,len(pos_vals))),float(sum(neg_vals)/max(1,len(neg_vals))),0.0)
