from __future__ import annotations
import numpy as np
from .classes import NUM_CLASSES, IGNORE_INDEX, CLASS_NAMES


def confusion_matrix(prediction, target, classes=NUM_CLASSES):
    p=np.asarray(prediction).astype(np.int64); t=np.asarray(target).astype(np.int64)
    if p.shape != t.shape: raise ValueError('Prediction/target shapes differ')
    valid=(t!=IGNORE_INDEX)
    if np.any((t[valid]<0)|(t[valid]>=classes)) or np.any((p[valid]<0)|(p[valid]>=classes)):
        raise ValueError('Invalid labels in confusion matrix')
    return np.bincount(classes*t[valid]+p[valid],minlength=classes*classes).reshape(classes,classes)


def metrics_from_confusion(cm):
    cm=np.asarray(cm,dtype=np.int64)
    tp=np.diag(cm).astype(float); support=cm.sum(1); predicted=cm.sum(0)
    union=support+predicted-tp
    iou=np.divide(tp,union,out=np.full(len(tp),np.nan),where=union>0)
    dice=np.divide(2*tp,support+predicted,out=np.full(len(tp),np.nan),where=(support+predicted)>0)
    def mean(a): return float(np.nanmean(a)) if np.isfinite(a).any() else None
    def div(a,b): return float(a/b) if b else None
    # Merge all disease classes. A disease-class confusion is still a correct lesion pixel.
    d_tp=int(cm[1:,1:].sum()); fp=int(cm[0,1:].sum()); fn=int(cm[1:,0].sum())
    total=int(cm.sum())
    metrics={
        'miou_all':mean(iou),'miou_foreground':mean(iou[1:]),
        'mean_dice_foreground':mean(dice[1:]),'pixel_accuracy':div(tp.sum(),total),
        'disease_iou':div(d_tp,d_tp+fp+fn),'disease_dice':div(2*d_tp,2*d_tp+fp+fn),
        'disease_pixel_recall':div(d_tp,d_tp+fn),'disease_pixel_precision':div(d_tp,d_tp+fp),
        'disease_false_positive_pixels':fp,'valid_pixels':total,
        'foreground_classes_with_gt':int(np.count_nonzero(support[1:])),
        'foreground_classes_with_union':int(np.count_nonzero(union[1:])),
        'definition':'Macro scores average over classes with nonzero GT-or-prediction union. Ignore=255 excluded. Binary disease merges labels 1..115. Scores are 0..1.'}
    perclass=[]
    for i in range(len(tp)):
        perclass.append({'id':i,'name':CLASS_NAMES[i] if len(tp)==NUM_CLASSES else str(i),
                         'iou':None if not np.isfinite(iou[i]) else float(iou[i]),
                         'dice':None if not np.isfinite(dice[i]) else float(dice[i]),
                         'gt_pixels':int(support[i]),'predicted_pixels':int(predicted[i])})
    return metrics,perclass
