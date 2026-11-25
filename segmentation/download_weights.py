#!/usr/bin/env python
"""Download official reusable SegFormer weights; these are NOT PlantSeg-trained."""
from __future__ import annotations
import argparse
from pathlib import Path
from plantseg_seg.io import load_config,write_json,file_hash,read_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/segformer_b2.yaml')
    p.add_argument('--revision',default='main')
    a=p.parse_args(); cfg=load_config(a.config)
    from huggingface_hub import HfApi,hf_hub_download
    model_id=cfg['model']['hub_id']; out=Path(cfg['model']['pretrained_dir']).resolve()
    if model_id not in {'nvidia/segformer-b2-finetuned-ade-512-512','nvidia/segformer-b0-finetuned-ade-512-512','nvidia/segformer-b1-finetuned-ade-512-512'}:
        raise ValueError('This downloader accepts only the three documented NVIDIA models')
    if (out/'provenance.json').exists():
        old=read_json(out/'provenance.json')
        if old['repo_id']!=model_id: raise ValueError('Existing pretrained directory belongs to a different model')
        if all((out/name).is_file() and file_hash(out/name)==h for name,h in old['sha256'].items()):
            print(f'Already downloaded and hash-verified: {out}'); return
        raise ValueError('Existing weight folder has missing/changed files. Use a fresh pretrained_dir or restore the files.')
    info=HfApi().model_info(model_id,revision=a.revision)
    revision=info.sha
    files={s.rfilename for s in info.siblings}
    weight='model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin'
    if weight not in files: raise FileNotFoundError(f'No supported unsharded weight file in {model_id}')
    names=['config.json',weight]+[n for n in ['preprocessor_config.json','README.md'] if n in files]
    out.mkdir(parents=True,exist_ok=True)
    for name in names:
        print(f'Downloading {model_id}@{revision}: {name}',flush=True)
        hf_hub_download(repo_id=model_id,filename=name,revision=revision,local_dir=str(out))
    write_json(out/'provenance.json',{'repo_id':model_id,'revision':revision,
        'sha256':{name:file_hash(out/name) for name in names},
        'pretraining':'ImageNet encoder + ADE20K semantic segmentation. NOT trained on PlantSeg.',
        'license':'See NVIDIA SegFormer source/model license; research/non-commercial restrictions may apply.',
        'source_url':f'https://huggingface.co/{model_id}'})
    print(f'Weights saved: {out}\nThe PlantSeg classifier will be initialized and fine-tuned by train_segmentation.py.')

if __name__=='__main__': main()
