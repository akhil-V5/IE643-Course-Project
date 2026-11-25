"""Visible grayscale masks are separate from lossless raw class-ID masks.
White=disease, black=background, mid-gray=ignored GT; never normalize raw IDs.
"""
from __future__ import annotations
from pathlib import Path
import textwrap
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .classes import NUM_CLASSES,IGNORE_INDEX,palette


def colored(labels):return Image.fromarray(palette()[np.asarray(labels,dtype=np.uint8)])


def disease_gray(labels):
    a=np.asarray(labels);out=np.zeros(a.shape,np.uint8)
    out[(a>0)&(a<NUM_CLASSES)]=255;out[a==IGNORE_INDEX]=127
    return Image.fromarray(out)


def overlay(image,labels,alpha=0.45):
    a=np.array(image.convert('RGB'),dtype=np.float32)
    if labels.shape!=a.shape[:2]:raise ValueError('Overlay must be aligned')
    keep=(labels>0)&(labels<NUM_CLASSES);colors=palette()[labels.astype(np.uint8)]
    a[keep]=a[keep]*(1-alpha)+colors[keep]*alpha
    return Image.fromarray(a.clip(0,255).round().astype(np.uint8))


def panel_grid(images,titles,path,max_side=320,columns=3):
    """Preview RGB with bilinear, mask panels with nearest-neighbor (by title)."""
    if len(images)!=len(titles) or not images:raise ValueError('One title per image')
    thumbs=[]
    for im,title in zip(images,titles):
        q=im.convert('RGB').copy()
        method=Image.Resampling.NEAREST if any(k in title.lower() for k in ('mask','classes','ids')) else Image.Resampling.BILINEAR
        scale=min(max_side/q.width,max_side/q.height)
        q=q.resize((max(1,round(q.width*scale)),max(1,round(q.height*scale))),method);thumbs.append(q)
    cw=max(max_side,220);ch=max(q.height for q in thumbs)+58
    cols=min(columns,len(thumbs));rows=(len(thumbs)+cols-1)//cols
    canvas=Image.new('RGB',(cols*cw,rows*ch+26),'white');d=ImageDraw.Draw(canvas)
    for i,(q,t) in enumerate(zip(thumbs,titles)):
        x=(i%cols)*cw;y=(i//cols)*ch
        native=f' [{images[i].width}x{images[i].height}]'
        d.multiline_text((x+6,y+5),'\n'.join(textwrap.wrap(t+native,45)),fill='black',spacing=3)
        px=x+(cw-q.width)//2;py=y+48
        canvas.paste(q,(px,py))
        # A black/white mask can be uniformly white: outline it so it cannot
        # disappear against the panel background. RGB/masks are display resized only.
        d.rectangle((px,py,px+q.width-1,py+q.height-1),outline=(100,100,100),width=1)
    d.text((6,rows*ch+5),'White=disease; black=background; gray=ignored GT. Previews resized; titles show native sizes.',fill='black')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def save_panel(image,predicted,ground_truth,path,max_side=320):
    ims=[image.copy()];titles=['Input RGB (not a mask)']
    if ground_truth is not None:
        ims.extend([disease_gray(ground_truth),colored(ground_truth),overlay(image,ground_truth)])
        titles.extend(['GROUND TRUTH disease mask','GROUND TRUTH classes','GROUND TRUTH overlay'])
    if predicted is not None:
        ims.extend([disease_gray(predicted),colored(predicted),overlay(image,predicted)])
        titles.extend(['PREDICTED disease mask','PREDICTED classes','PREDICTED overlay'])
    panel_grid(ims,titles,path,max_side)


def save_prediction(image,labels,probability,out,key,preview=True):
    out=Path(out);paths={}
    items={'class_ids':Image.fromarray(labels.astype(np.uint8)),
           'class_colors':colored(labels),'disease_binary':disease_gray(labels),'overlays':overlay(image,labels)}
    if probability is not None:
        items['probability']=Image.fromarray(np.round(np.clip(probability,0,1)*65535).astype(np.uint16))
    for folder,im in items.items():
        p=out/folder/(key+'.png');p.parent.mkdir(parents=True,exist_ok=True);im.save(p);paths[folder]=str(p.resolve())
    if preview:save_panel(image,labels,None,out/'preview'/(key+'.png'))
    return paths


def save_ground_truth(image,labels,out,key):
    out=Path(out);items={'gt_class_ids':Image.fromarray(labels.astype(np.uint8)),
          'gt_disease_binary':disease_gray(labels),'gt_class_colors':colored(labels),'gt_overlays':overlay(image,labels),
          'gt_valid':Image.fromarray(((labels!=255)*255).astype(np.uint8))}
    for folder,im in items.items():
        p=out/folder/(key+'.png');p.parent.mkdir(parents=True,exist_ok=True);im.save(p)
