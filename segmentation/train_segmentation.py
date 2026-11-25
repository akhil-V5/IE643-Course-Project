#!/usr/bin/env python
"""Fine-tune existing SegFormer weights on original multiclass PlantSeg masks."""
from __future__ import annotations
import argparse
from pathlib import Path
from plantseg_seg.io import load_config,safe_load
from plantseg_seg.engine import run_training


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/segformer_b2.yaml')
    p.add_argument('--run',required=True)
    p.add_argument('--prepared',help='Use the ONE shared audited dataset')
    p.add_argument('--preview-count',type=int,help='0 saves every validation panel; individual masks are always saved by default')
    p.add_argument('--resume',help='Continue model + optimizer + sampler from last.pth; --steps is TOTAL target updates')
    p.add_argument('--init',help='Use PlantSeg-trained weights for a NEW optimizer/schedule/run')
    stop=p.add_mutually_exclusive_group()
    stop.add_argument('--steps',type=int); stop.add_argument('--epochs',type=int)
    p.add_argument('--batch-size',type=int); p.add_argument('--accumulate',type=int)
    p.add_argument('--workers',type=int); p.add_argument('--image-size',type=int)
    p.add_argument('--val-limit',type=int); p.add_argument('--save-every',type=int)
    p.add_argument('--validate-every',type=int); p.add_argument('--warmup',type=int)
    p.add_argument('--trainable',choices=['all','decoder'])
    p.add_argument('--device',choices=['cuda','cpu'])
    p.add_argument('--no-amp',action='store_true')
    a=p.parse_args()
    cfg=safe_load(a.resume)['config'] if a.resume else load_config(a.config)
    for arg,key in [('batch_size','batch_size'),('accumulate','accumulate'),('val_limit','val_limit'),
                    ('save_every','save_every'),('validate_every','validate_every'),('warmup','warmup'),
                    ('trainable','trainable'),('device','device')]:
        v=getattr(a,arg)
        if v is not None: cfg['train'][key]=v
    if a.steps is not None:
        if a.steps<1: raise ValueError('--steps must be positive')
        cfg['train']['steps']=a.steps
    if a.epochs is not None:
        if a.epochs<1: raise ValueError('--epochs must be positive')
        cfg['train']['epochs']=a.epochs; cfg['train']['steps']=None
    if a.prepared: cfg['data']['prepared']=str(Path(a.prepared).resolve())
    if a.preview_count is not None: cfg['train']['preview_count']=a.preview_count
    if a.workers is not None: cfg['data']['workers']=a.workers
    if a.image_size is not None: cfg['data']['image_size']=a.image_size
    if a.no_amp: cfg['train']['amp']=False
    run_training(cfg,a.run,resume=a.resume,init=a.init)

if __name__=='__main__': main()
