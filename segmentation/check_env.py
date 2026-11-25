#!/usr/bin/env python
import argparse
import sys
import importlib.metadata
from pathlib import Path
import torch
from plantseg_seg.io import load_config,choose_device
from plantseg_seg.model import initialize_pretrained
from plantseg_seg.losses import segmentation_loss
from plantseg_seg.classes import NUM_CLASSES


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/segformer_b2.yaml')
    p.add_argument('--check-model',action='store_true',help='Load real downloaded checkpoint; forward/backward test at 128px')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args()
    print('Python:',sys.version,'\nInterpreter:',sys.executable)
    for name in ('torch','torchvision','transformers','huggingface-hub','safetensors','numpy','Pillow'):
        try: print(name,importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError: print(name,'NOT INSTALLED')
    print('CUDA:',torch.cuda.is_available())
    if torch.cuda.is_available(): print('GPU:',torch.cuda.get_device_name(0))
    if a.check_model:
        device=choose_device(a.device); cfg=load_config(a.config)
        model,provenance=initialize_pretrained(cfg['model']['pretrained_dir']); model.to(device).train()
        x=torch.randn(1,3,128,128,device=device)
        y=torch.zeros(1,128,128,dtype=torch.long,device=device); y[:,40:70,50:90]=3
        logits=model(pixel_values=x).logits
        loss,_=segmentation_loss(logits,y)
        loss.backward()
        grad=model.decode_head.classifier.weight.grad
        assert logits.shape[1]==NUM_CLASSES and torch.isfinite(loss) and torch.isfinite(grad).all()
        assert grad.abs().sum()>0
        print('PASS: real pretrained model loaded, 116-class logits produced, and finite gradients reached new classifier.')
        print('This is a mechanics test, not evidence of disease-localization accuracy.')

if __name__=='__main__': main()
