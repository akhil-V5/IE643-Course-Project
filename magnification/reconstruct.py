"""Reconstruct the EXACT shared cohort or arbitrary inputs. No lesion annotations
are passed to ResShift. Finetuned and pretrained inference both use frozen weights.
"""
from __future__ import annotations
import argparse,hashlib,time
from pathlib import Path
import numpy as np
from PIL import Image
from common.runtime import *
from experiments.cases import verify_cases
from experiments.metrics import rgb_metrics,aggregate_rgb
from plantseg_seg.data import load_rgb,load_labels
from plantseg_seg.visuals import panel_grid,disease_gray,save_ground_truth
EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}


def collect_inputs(path):
    p=Path(path).resolve()
    if not p.exists():raise FileNotFoundError(p)
    return [p] if p.is_file() else sorted(x for x in p.rglob('*') if x.is_file() and x.suffix.lower() in EXTS)


def reconstruct(config,output,comparison=None,input_path=None,checkpoint=None,backend_override=None):
    import copy
    if bool(comparison)==bool(input_path):raise ValueError('Choose comparison OR input, not both')
    cfg=copy.deepcopy(load_yaml(config) if isinstance(config,(str,Path)) else config)
    mode=cfg['inference']['method'];scale=int(cfg['inference']['scale'])
    if mode=='resshift':
        if int(cfg['inference'].get('tile_lr',64)) != 64:
            raise ValueError('The restored v1 tiled wrapper uses tile_lr=64')
        overlap = int(cfg['inference'].get('overlap',16))
        if not 0 <= overlap < 64:
            raise ValueError('overlap must be 0..63 LR pixels')
        cfg['inference'].update(sampling_mode='tiled', tile_lr=64, overlap=overlap)
    if mode not in ('resshift','bicubic') or scale!=4:raise ValueError('Supported methods: x4 ResShift or bicubic')
    if mode=='bicubic' and checkpoint:raise ValueError('Bicubic has no checkpoint')
    meta=None;root=Path(comparison).resolve() if comparison else None
    if root:
        meta,rows=verify_cases(root)
    else:
        files=collect_inputs(input_path)
        if not files:raise ValueError('No images found')
        dest=Path(output).resolve();ip=Path(input_path).resolve()
        if ip.is_dir() and (dest==ip or dest.is_relative_to(ip)):raise ValueError('Output must be outside input directory; prevent reprocessing prior previews')
        rows=[{'id':hashlib.sha256(str(p).encode()).hexdigest()[:20],'input':str(p)} for p in files]
    backend=None;selected=None;checkpoint_sha=None
    if mode=='resshift':
        from leafdiff.backend import ResShiftBackend
        from leafdiff.common import safe_load
        import torch
        selected=checkpoint or cfg['backend']['diffusion_weights'];checkpoint_sha=digest(selected)
        if checkpoint and backend_override is None:
            saved=safe_load(checkpoint)
            if 'config' not in saved or 'manifest_sha256' not in saved or int(saved.get('step',0))<1:
                raise ValueError('Finetuned condition requires a trained checkpoint with shared-split provenance; do not silently label pretrained weights finetuned')
            if meta and saved['manifest_sha256']!=meta['source_manifest_sha256']:
                raise ValueError('ResShift was trained with a DIFFERENT source split manifest')
            del saved
        backend=backend_override or ResShiftBackend(cfg,checkpoint=selected,device=cfg['inference']['device'])
        if hasattr(backend,'model'):backend.model.eval().requires_grad_(False)
        if hasattr(backend,'ae'):backend.ae.eval().requires_grad_(False)
    out=fresh_dir(output);record=[];scores=[];preview_count=int(cfg['inference'].get('preview_count',12))
    write_json(out/'resolved_config.json',cfg)
    print(f'Reconstruction: {len(rows)} images; optional visual examples={preview_count} (0=all). RGB outputs for ALL selected images are required for the next segmentation stage.',flush=True)
    for i,r in enumerate(rows):
        ip=(root/r['lr']) if root else Path(r['input']);image=load_rgb(ip)
        seed=(int(cfg['inference']['seed'])+int(hashlib.sha256(r['id'].encode()).hexdigest()[:8],16))%(2**31)
        start=time.perf_counter()
        if mode=='bicubic':
            restored=image.resize((image.width*4,image.height*4),Image.Resampling.BICUBIC)
            sampling={'sampling_mode':'bicubic','sample_calls':0}
        else:
            from leafdiff.tiling import tiled_reconstruct
            restored,sampling=tiled_reconstruct(backend,image,seed=seed,
                       overlap=int(cfg['inference']['overlap']),amp=bool(cfg['inference']['amp']))
            if backend.device.type=='cuda':
                import torch;torch.cuda.synchronize(backend.device)
        seconds=time.perf_counter()-start
        if restored.size!=(image.width*4,image.height*4):raise ValueError(f'Unexpected reconstruction dimensions for {r["id"]}')
        if root:
            if restored.size!=(int(r['padded_width']),int(r['padded_height'])):raise ValueError('Reconstruction is not on declared padded x4 grid')
            restored=restored.crop((0,0,int(r['width']),int(r['height'])))
        dest=out/'rgb'/(r['id']+'.png');dest.parent.mkdir(parents=True,exist_ok=True);restored.save(dest)
        entry={'id':r['id'],'rgb':str(dest.relative_to(out)),'rgb_sha256':digest(dest),'width':restored.width,
             'height':restored.height,'input':str(ip),'input_sha256':digest(ip),**sampling,'seed':seed,'seconds':seconds}
        if root:
            original=load_rgb(root/r['reference']);gt=load_labels(root/r['mask']);bic=load_rgb(root/r['bicubic'])
            values=rgb_metrics(restored,original);entry.update(values);scores.append(values)
            if preview_count==0 or i<preview_count:
                save_ground_truth(original,gt,out,r['id'])
                panel_grid([original,image,bic,restored,disease_gray(gt)],['Original reference RGB','Actual low-resolution RGB (preview)',
                           'Bicubic RGB','Reconstructed RGB: '+mode,'GROUND TRUTH disease mask (not prediction)'],out/'panels'/(r['id']+'.png'))
        elif preview_count==0 or i<preview_count:
            panel_grid([image,restored],['Input RGB','Reconstructed RGB: '+mode],out/'panels'/(r['id']+'.png'))
        record.append(entry);append_json(out/'progress.jsonl',entry)
        print(f'Reconstructed {i+1}/{len(rows)} {r["id"]}: {image.size} -> {restored.size}; sampling_mode={sampling["sampling_mode"]}; sample_calls={sampling["sample_calls"]}; {seconds:.3f}s',flush=True)
    write_csv(out/'images.csv',record)
    info={'schema':'plantseg_reconstruction_v3','method':mode,'condition':'finetuned' if checkpoint else ('pretrained' if mode=='resshift' else 'bicubic'),
       'sampling_mode':'tiled' if mode=='resshift' else 'bicubic',
       'actual_backend':backend_override is None,'checkpoint':str(Path(selected).resolve()) if selected else None,
       'checkpoint_sha256':checkpoint_sha,'images_csv_sha256':digest(out/'images.csv'),
       'cases_sha256':meta['cases_sha256'] if meta else None,'source_manifest_sha256':meta['source_manifest_sha256'] if meta else None,
       'split':meta['split'] if meta else None,'images':len(record),'inference_config':cfg['inference'],'versions':versions(),
       'label_inputs':'None. Annotations are read for integrity checks and output scoring/visualization, never passed to the reconstruction model.',
       'metrics':aggregate_rgb(scores) if meta else None,
       'metric_note':'RGB PSNR/SSIM on original unpadded image; no GT-dependent pixel selection. Without references these metrics are unavailable.'}
    write_json(out/'manifest.json',info)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--comparison');g.add_argument('--input')
    p.add_argument('--checkpoint');a=p.parse_args();reconstruct(a.config,a.output,a.comparison,a.input,a.checkpoint)
if __name__=='__main__':main()
