"""Evaluate all requested image conditions using ONE frozen trained segmenter,
ONE set of original masks, ONE output grid, and ONE preprocessing policy.
"""
from __future__ import annotations
import argparse,time
from pathlib import Path
import numpy as np
from common.runtime import *
from experiments.cases import verify_cases
from experiments.metrics import rgb_metrics,aggregate_rgb
from plantseg_seg.data import load_rgb,load_labels
from plantseg_seg.classes import NUM_CLASSES
from plantseg_seg.predictor import PlantSegSegmenter
from plantseg_seg.metrics import confusion_matrix,metrics_from_confusion
from plantseg_seg.visuals import save_prediction,save_ground_truth,panel_grid,disease_gray,overlay


def load_variant(path,metadata,case_rows):
    p=Path(path).resolve();info=read_json(p);folder=p.parent
    if info['schema']!='plantseg_reconstruction_v3':raise ValueError('Expected reconstruction manifest from magnification/reconstruct.py')
    if info.get('method') == 'resshift' and info.get('sampling_mode') not in {'tiled', 'whole_image'}:
        raise ValueError('ResShift output lacks valid sampling metadata; regenerate it.')
    if info['cases_sha256']!=metadata['cases_sha256'] or info['source_manifest_sha256']!=metadata['source_manifest_sha256']:
        raise ValueError('Reconstruction condition uses DIFFERENT cases or source split')
    validate_file(folder/'images.csv',info['images_csv_sha256']);rows=read_csv(folder/'images.csv')
    if [r['id'] for r in rows]!=[r['id'] for r in case_rows]:raise ValueError('Missing, extra, reordered or duplicate reconstructed cases')
    mapping={}
    for r,c in zip(rows,case_rows):
        img=folder/r['rgb'];validate_file(img,r['rgb_sha256'])
        if (int(r['width']),int(r['height']))!=(int(c['width']),int(c['height'])):raise ValueError('Reconstruction dimensions mismatch')
        if r['input_sha256']!=c['lr_sha256']:raise ValueError('Reconstruction did not use exactly the shared LR input')
        mapping[r['id']]=img
    return info,mapping


def evaluate(checkpoint,comparison,output,modes,variants=None,device='cuda',image_size=None,preview_count=12,segmenter_override=None,amp=True):
    if not modes or not set(modes)<={'original','bicubic','resshift-pretrained','resshift-finetuned'} or len(set(modes))!=len(modes):raise ValueError('Use unique supported evaluation conditions')
    root=Path(comparison).resolve();meta,rows=verify_cases(root)
    segmenter=segmenter_override or PlantSegSegmenter(checkpoint,device,image_size,amp)
    ck=segmenter.checkpoint
    if ck['manifest_sha256']!=meta['source_manifest_sha256']:raise ValueError('SegFormer checkpoint used DIFFERENT split manifest')
    variants=variants or {};paths={};variant_info={}
    for mode in modes:
        if mode in ('original','bicubic'):paths[mode]={r['id']:root/r['reference' if mode=='original' else 'bicubic'] for r in rows}
        elif mode in variants:
            variant_info[mode],paths[mode]=load_variant(variants[mode],meta,rows)
            expected='finetuned' if mode=='resshift-finetuned' else 'pretrained'
            if variant_info[mode]['method']!='resshift' or variant_info[mode]['condition']!=expected:raise ValueError('Reconstruction manifest is mislabeled: '+mode)
        else:raise ValueError(f'No reconstruction manifest supplied for {mode}')
    out=fresh_dir(output);summary=[];image_details=[];input_artifacts=[]
    for mode in modes:
        folder=out/mode;cm=np.zeros((NUM_CLASSES,NUM_CLASSES),np.int64);rgb=[];details=[]
        for i,r in enumerate(rows):
            reference=load_rgb(root/r['reference']);target=load_labels(root/r['mask']);image=load_rgb(paths[mode][r['id']])
            if image.size!=reference.size:raise ValueError('Do not resize mismatched reconstructions to hide alignment errors')
            start=time.perf_counter();prediction=segmenter.predict(image)  # never given target/GT
            seconds=time.perf_counter()-start
            current=confusion_matrix(prediction.class_ids,target);cm+=current;one,_=metrics_from_confusion(current)
            values=rgb_metrics(image,reference) if mode!='original' else {'psnr_rgb_db':None,'ssim_rgb':None}
            if mode!='original':rgb.append(values)
            d={'id':r['id'],'mode':mode,'seconds_segmentation':seconds,**values,'miou_foreground':one['miou_foreground'],'disease_dice':one['disease_dice']}
            details.append(d);image_details.append(d);append_json(folder/'progress.jsonl',d)
            if preview_count==0 or i<preview_count:
                save_prediction(
                    image, prediction.class_ids,
                    prediction.disease_probability,
                    folder, r['id'], preview=False
                )
                save_ground_truth(reference,target,folder,r['id'])
                panel_grid([reference,image,disease_gray(target),disease_gray(prediction.class_ids),overlay(reference,target),overlay(image,prediction.class_ids)],
                    ['Original RGB reference','SegFormer input: '+mode,'GROUND TRUTH disease mask','PREDICTED disease mask','GROUND TRUTH overlay','PREDICTED overlay'],folder/'panels'/(r['id']+'.png'))
            input_artifacts.append({'id':r['id'],'mode':mode,'input':str(paths[mode][r['id']]),'input_sha256':digest(paths[mode][r['id']])})
            if (i+1)%25==0 or i+1==len(rows):print(f'{mode}: segmented {i+1}/{len(rows)}',flush=True)
        metrics,classes=metrics_from_confusion(cm);agg=aggregate_rgb(rgb)
        result={'condition':mode,'images':len(rows),'PSNR_dB':agg['PSNR_dB'],'SSIM':agg['SSIM'],
                'foreground_multiclass_mIoU':metrics['miou_foreground'],'binary_disease_Dice':metrics['disease_dice']}
        summary.append(result);write_json(folder/'metrics.json',{'main':result,'rgb_details':agg,'segmentation_details':metrics})
        write_csv(folder/'per_image.csv',details);write_csv(folder/'per_class.csv',classes);np.save(folder/'confusion_matrix.npy',cm)
        print(result,flush=True)
    # Row-wise montage across conditions uses the same representative images in all branches.
    for i,r in enumerate(rows):
        if preview_count and i>=preview_count:break
        ims=[load_rgb(root/r['reference']),disease_gray(load_labels(root/r['mask']))];titles=['Original RGB reference','GROUND TRUTH disease mask']
        for mode in modes:
            from PIL import Image
            with Image.open(out/mode/'disease_binary'/(r['id']+'.png')) as im:pred=im.copy()
            ims.extend([load_rgb(paths[mode][r['id']]),pred]);titles.extend([mode+' RGB',mode+' predicted disease mask'])
        panel_grid(ims,titles,out/'comparison_panels'/(r['id']+'.png'),columns=2)
    write_csv(out/'summary.csv',summary);write_csv(out/'per_image_all_conditions.csv',image_details);write_csv(out/'input_artifacts.csv',input_artifacts)
    policy={'segmentation':'Pooled original-grid valid-pixel confusion. Foreground mIoU averages classes1..115 with GT-or-prediction union>0. Background excluded ONLY from averaging, not false positives. Binary Dice merges all disease labels. Ignore255 excluded; undefined masks return null.',
            'rgb':'Unpadded RGB [0,1], no border shaving. Per-image PSNR and SSIM then mean images. SSIM window7 (smaller odd window when necessary); null below3 pixels. Identical RGB PSNR=Infinity. Original RGB row has null reconstruction scores.',
            'preprocessing':'All conditions letterboxed using SAME segmenter canvas/normalization, logits restored before argmax; original GT untouched.',
            'padding':'Edge pad RGB to multiple4 only to form LR; unpad reconstruction before segmentation/scoring. No image dropped.',
            'score_scale':'mIoU and Dice stored in [0,1], multiply by100 only for reporting percentages.',
            'ground_truth_access':'Annotation used only for loss/training or evaluation/visualization, NEVER as an inference input.'}
    write_json(out/'summary.json',{'results':summary,'checkpoint':str(Path(checkpoint).resolve()),'checkpoint_sha256':digest(checkpoint),
        'training_updates':ck['step'],'source_manifest_sha256':meta['source_manifest_sha256'],'cases_sha256':meta['cases_sha256'],
        'split':meta['split'],'subset':bool(meta['limit']),'segmenter_canvas':segmenter.image_size,'amp':bool(amp),
        'actual_segformer_used':segmenter_override is None,'variant_provenance':variant_info,'metric_policy':policy,'versions':versions()})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--comparison',required=True);p.add_argument('--output',required=True)
    p.add_argument('--modes',nargs='+',required=True);p.add_argument('--variant',action='append',default=[],help='condition=/path/to/manifest.json')
    p.add_argument('--device',default='cuda');p.add_argument('--image-size',type=int);p.add_argument('--preview-count',type=int,default=12);p.add_argument('--no-amp',action='store_true')
    a=p.parse_args();variants=dict(s.split('=',1) for s in a.variant)
    evaluate(a.checkpoint,a.comparison,a.output,a.modes,variants,a.device,a.image_size,a.preview_count,amp=not a.no_amp)
if __name__=='__main__':main()
