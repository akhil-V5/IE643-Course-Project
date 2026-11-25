"""Standalone segmentation of original/bicubic/ResShift RGB. No GT required.
Metrics are available only when a correctly aligned --mask-dir is supplied.
For benchmark metrics prefer run.py evaluate, which enforces split identity.
"""
from __future__ import annotations
import argparse,hashlib,time
from pathlib import Path
import numpy as np
from common.runtime import *
from plantseg_seg.data import load_rgb,load_labels
from plantseg_seg.predictor import PlantSegSegmenter,class_summary
from plantseg_seg.visuals import save_prediction,save_ground_truth,save_panel
from plantseg_seg.classes import NUM_CLASSES
from plantseg_seg.metrics import confusion_matrix,metrics_from_confusion
from magnification.reconstruct import collect_inputs


def infer(checkpoint,output,input_path=None,reconstruction_manifest=None,device='cuda',image_size=None,mask_dir=None,preview_count=12,amp=True,segmenter_override=None):
    if bool(input_path)==bool(reconstruction_manifest):raise ValueError('Choose input or reconstruction manifest')
    if reconstruction_manifest:
        p=Path(reconstruction_manifest).resolve();info=read_json(p);validate_file(p.parent/'images.csv',info['images_csv_sha256'])
        src=read_csv(p.parent/'images.csv');inputs=[]
        for r in src:
            f=p.parent/r['rgb'];validate_file(f,r['rgb_sha256']);inputs.append((r['id'],f))
    else:
        p=Path(input_path).resolve();outp=Path(output).resolve()
        if p.is_dir() and (outp==p or outp.is_relative_to(p)):raise ValueError('Output must be outside input tree')
        inputs=[(hashlib.sha256(str(f).encode()).hexdigest()[:20],f) for f in collect_inputs(p)]
    if not inputs:raise ValueError('No input RGB files')
    if mask_dir:
        # Explicitly one mask per source stem; ambiguous basenames are rejected.
        stems=[f.stem for _,f in inputs]
        if len(stems)!=len(set(stems)):raise ValueError('Duplicate input stems; use shared-cohort evaluation for unique matching')
        for _,f in inputs:
            if not (Path(mask_dir)/(f.stem+'.png')).is_file():raise FileNotFoundError(f'Missing annotation for {f}')
    segmenter=segmenter_override or PlantSegSegmenter(checkpoint,device,image_size,amp)
    out=fresh_dir(output);cm=np.zeros((NUM_CLASSES,NUM_CLASSES),np.int64);records=[]
    for i,(uid,f) in enumerate(inputs):
        image=load_rgb(f);start=time.perf_counter();pred=segmenter.predict(image);elapsed=time.perf_counter()-start
        outputs=save_prediction(image,pred.class_ids,pred.disease_probability,out,uid,preview=False)
        target=None
        if mask_dir:
            target=load_labels(Path(mask_dir)/(f.stem+'.png'))
            if target.shape!=(image.height,image.width):raise ValueError('GT does not match supplied RGB dimensions; do not resize GT')
            cm+=confusion_matrix(pred.class_ids,target);save_ground_truth(image,target,out,uid)
        if preview_count==0 or i<preview_count:save_panel(image,pred.class_ids,target,out/'panels'/(uid+'.png'))
        record={'id':uid,'input':str(f),'input_sha256':digest(f),'width':image.width,'height':image.height,'seconds':elapsed,
                'class_ids':outputs['class_ids'],'disease_binary':outputs['disease_binary']}
        records.append(record);append_json(out/'progress.jsonl',record)
        write_json(out/'classes'/(uid+'.json'),class_summary(pred.class_ids))
        print(f'Predicted {i+1}/{len(inputs)}: {f.name}',flush=True)
    write_csv(out/'images.csv',records)
    metrics=None
    if mask_dir:
        m,classes=metrics_from_confusion(cm);metrics={'foreground_multiclass_mIoU':m['miou_foreground'],'binary_disease_Dice':m['disease_dice']}
        write_json(out/'metrics.json',{'main':metrics,'details':m,'benchmark_warning':'Ad-hoc annotations; this mode does not establish a held-out PlantSeg split.'})
        write_csv(out/'per_class.csv',classes);np.save(out/'confusion_matrix.npy',cm)
    write_json(out/'inference.json',{'schema':'plantseg_inference_v3','checkpoint':str(Path(checkpoint).resolve()),'checkpoint_sha256':digest(checkpoint),
        'input_canvas':segmenter.image_size,'amp':amp,'images':len(records),'metrics':metrics,
        'note':'No annotation supplied => metrics cannot be calculated. Predictions are not proof of biological health.',
        'actual_segformer_used':segmenter_override is None,'versions':versions()})
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--input');g.add_argument('--reconstruction-manifest')
    p.add_argument('--device',default='cuda');p.add_argument('--image-size',type=int);p.add_argument('--mask-dir');p.add_argument('--preview-count',type=int,default=12);p.add_argument('--no-amp',action='store_true')
    a=p.parse_args();infer(a.checkpoint,a.output,a.input,a.reconstruction_manifest,a.device,a.image_size,a.mask_dir,a.preview_count,not a.no_amp)
if __name__=='__main__':main()
