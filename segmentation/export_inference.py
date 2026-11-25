#!/usr/bin/env python
"""Strip optimizer/scaler/RNG state from a trusted trained checkpoint for deployment."""
from pathlib import Path
import argparse
from plantseg_seg.io import safe_load, atomic_torch_save
from plantseg_seg.classes import CLASS_HASH


def export(source,output):
    c=safe_load(source)
    if c.get('task')!='plantseg_multiclass_v1' or c.get('class_hash')!=CLASS_HASH or c.get('step',0)<1:
        raise ValueError('Expected a trained PlantSeg checkpoint')
    path=Path(output)
    if path.exists():raise FileExistsError(path)
    fields=['task','class_hash','model','model_config','config','step','initialization','manifest_sha256']
    light={key:c[key] for key in fields};light['inference_only']=True
    atomic_torch_save(light,path)
    print(f'Saved inference-only checkpoint: {path}. Use the original checkpoint to resume training.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();export(a.checkpoint,a.output)
