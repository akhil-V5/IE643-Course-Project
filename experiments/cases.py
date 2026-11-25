"""One immutable whole-image cohort shared by original/bicubic/ResShift conditions.
Never drops source images. RGB is edge-padded to the x4 grid only for degradation;
reconstructions are unpadded before segmentation and RGB scoring. GT is untouched.
"""
from __future__ import annotations
import argparse, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from common.runtime import *
from plantseg_seg.audit import verify_prepared
from plantseg_seg.data import load_rgb,load_labels
from plantseg_seg.classes import CLASS_HASH
from plantseg_seg.visuals import save_ground_truth,panel_grid,disease_gray


def pad_rgb(image,scale=4):
    a=np.asarray(image.convert('RGB'));h,w=a.shape[:2]
    ph=(-h)%scale;pw=(-w)%scale
    return Image.fromarray(np.pad(a,((0,ph),(0,pw),(0,0)),mode='edge')),{'width':w,'height':h,'pad_right':pw,'pad_bottom':ph}


def verify_cases(folder,expected_manifest=None,verify_sources=True):
    root=Path(folder).resolve();m=read_json(root/'metadata.json')
    if m['schema']!='plantseg_cases_v3' or m['class_hash']!=CLASS_HASH:raise ValueError('Wrong case schema or class mapping')
    if digest(root/'cases.csv')!=m['cases_sha256']:raise ValueError('Comparison CSV changed')
    if expected_manifest and m['source_manifest_sha256']!=expected_manifest:raise ValueError('Different split manifest from segmentation checkpoint')
    meta,originals=verify_prepared(m['source_prepared'],expected_manifest=m['source_manifest_sha256'],verify_content=verify_sources)
    expected=[r for r in originals if r['split']==m['split']]
    if m['limit']:expected=expected[:m['limit']]
    rows=read_csv(root/'cases.csv')
    if [r['id'] for r in rows]!=[r['id'] for r in expected]:raise ValueError('Comparison source membership/order differs from declared split')
    for r,src in zip(rows,expected):
        if r['width']!=src['width'] or r['height']!=src['height']:raise ValueError('Case dimensions differ from originals')
        for key in ('reference','mask','lr','bicubic'):
            p=root/r[key];validate_file(p,r[key+'_sha256'])
        with Image.open(root/r['mask']) as im:
            if np.asarray(im).shape!=(int(r['height']),int(r['width'])):raise ValueError('GT mask was resized')
    return m,rows


def make_cases(prepared,output,split='test',scale=4,limit=0,preview_count=12):
    if split not in ('val','test') or scale!=4 or limit<0:raise ValueError('Use held-out val/test, x4, nonnegative limit')
    meta,allrows=verify_prepared(prepared,verify_content=True)
    selected=[r for r in allrows if r['split']==split]
    if limit:selected=selected[:limit]
    if not selected:raise ValueError('Empty split')
    out=Path(output).resolve()
    if (out/'metadata.json').exists():
        old,rows=verify_cases(out,meta['manifest_sha256'])
        if (old['split'],old['scale'],old['limit'])!=(split,scale,limit):raise ValueError('Cached comparison policy differs; choose another cache folder')
        print(f'Verified existing cohort: {len(rows)} images in {out}',flush=True);return out
    out=fresh_dir(out);records=[]
    for i,row in enumerate(selected):
        im=load_rgb(row['image']);gt=load_labels(row['mask']);padded,geometry=pad_rgb(im,scale)
        lr=padded.resize((padded.width//scale,padded.height//scale),Image.Resampling.BICUBIC)
        bic=lr.resize(padded.size,Image.Resampling.BICUBIC).crop((0,0,im.width,im.height))
        ims={'reference':im,'mask':Image.fromarray(gt),'lr':lr,'bicubic':bic};paths={}
        for key,img in ims.items():
            rel=Path(key)/(row['id']+'.png');p=out/rel;p.parent.mkdir(parents=True,exist_ok=True);img.save(p);paths[key]=rel.as_posix()
        records.append({'id':row['id'],'split':split,**geometry,
                        'padded_width':padded.width,'padded_height':padded.height,
                        'source_image_sha256_decoded':row['image_sha256_decoded'],
                        'source_mask_sha256_decoded':row['mask_sha256_decoded'],**paths,
                        **{k+'_sha256':digest(out/v) for k,v in paths.items()}})
        if preview_count==0 or i<preview_count:
            save_ground_truth(im,gt,out,row['id'])
            panel_grid([im,lr,bic,disease_gray(gt)],['Original reference RGB','Actual LR RGB (preview)','Bicubic x4 RGB','GROUND TRUTH disease mask'],out/'panels'/(row['id']+'.png'))
        if (i+1)%50==0 or i+1==len(selected):print(f'Prepared comparison {i+1}/{len(selected)}',flush=True)
    write_csv(out/'cases.csv',records)
    write_json(out/'metadata.json',{'schema':'plantseg_cases_v3','class_hash':CLASS_HASH,
        'source_prepared':str(Path(prepared).resolve()),'source_manifest_sha256':meta['manifest_sha256'],
        'cases_sha256':digest(out/'cases.csv'),'split':split,'scale':scale,'limit':limit,'samples':len(records),
        'source_split_sha256':meta['split_sha256'][split],'sampling':'all whole images; no disease-based selection; no cropping or exclusion',
        'padding':'replicate RGB right/bottom to multiple4 before downsampling; unpad outputs BEFORE segmenting and scoring',
        'degradation':'Pillow BICUBIC to 1/4 padded dimensions; fixed uint8 PNG inputs shared by every reconstruction method',
        'ground_truth':'Original indexed annotation retained EXACTLY; never resized for evaluation or passed into reconstruction',
        'versions':versions()})
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',required=True);p.add_argument('--output',required=True)
    p.add_argument('--split',choices=['val','test'],default='test');p.add_argument('--limit',type=int,default=0);p.add_argument('--preview-count',type=int,default=12)
    a=p.parse_args();make_cases(a.prepared,a.output,a.split,limit=a.limit,preview_count=a.preview_count)
if __name__=='__main__':main()
