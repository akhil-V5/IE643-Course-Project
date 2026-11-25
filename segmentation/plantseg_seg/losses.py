from __future__ import annotations
import torch
import torch.nn.functional as F
from .classes import IGNORE_INDEX


def segmentation_loss(logits, target, disease_dice_weight=0.0):
    """Multiclass cross entropy; optional binary-union Dice auxiliary loss.

    The optional Dice term never changes the 116-class head or target IDs.
    """
    logits=F.interpolate(logits.float(),size=target.shape[-2:],mode='bilinear',align_corners=False)
    valid=target!=IGNORE_INDEX
    if not valid.any(): raise ValueError('All-ignore minibatch')
    ce=F.cross_entropy(logits,target,ignore_index=IGNORE_INDEX)
    aux=logits.sum()*0
    if disease_dice_weight:
        p=1-logits.softmax(1)[:,0]
        t=(target>0)&valid
        p=p*valid
        inter=(p*t).sum(dim=(-2,-1)); denom=p.sum(dim=(-2,-1))+t.sum(dim=(-2,-1))
        # Smoothing protects all-negative targets; CE remains the main loss there.
        aux=(1-(2*inter+1.0)/(denom+1.0)).mean()
    total=ce+float(disease_dice_weight)*aux
    return total,{'cross_entropy':float(ce.detach()),'disease_dice_loss':float(aux.detach()),'total':float(total.detach())}
