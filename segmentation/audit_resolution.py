#!/usr/bin/env python
"""Report lesions/valid labels lost at the chosen training canvas. Never alter masks."""
from __future__ import annotations
import argparse
import numpy as np
from plantseg_seg.audit import verify_prepared
from plantseg_seg.data import load_rgb,load_labels
from plantseg_seg.geometry import letterbox
from plantseg_seg.io import write_csv
from plantseg_seg.classes import NUM_CLASSES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True);p.add_argument('--image-size',type=int,default=512)
    p.add_argument('--output',required=True);a=p.parse_args()
    _,rows=verify_prepared(a.prepared)
    report=[]
    for row in rows:
        image,labels=load_rgb(row['image']),load_labels(row['mask'])
        _,target,_=letterbox(image,labels,a.image_size)
        t=target.numpy();d=(labels>0)&(labels<NUM_CLASSES);small=(t>0)&(t<NUM_CLASSES)
        lost=sorted(set(np.unique(labels[d]).tolist())-set(np.unique(t[small]).tolist()))
        report.append({'id':row['id'],'split':row['split'],'image':row['image'],
            'native_disease_pixels':int(d.sum()),'resized_disease_pixels':int(small.sum()),
            'all_disease_lost':bool(d.any() and not small.any()),'lost_class_ids':','.join(map(str,lost)),
            'all_labels_ignored_after_resize':bool(np.all(t==255))})
    write_csv(a.output,report)
    print({'images':len(report),'all_disease_lost':sum(r['all_disease_lost'] for r in report),
           'some_disease_classes_lost':sum(bool(r['lost_class_ids']) for r in report)})

if __name__=='__main__':main()
