"""Reusable segmentation API for ORIGINAL, BICUBIC or ResShift RGB inputs.

No disease gate by default. Optional legacy screening is explicit, not a pipeline step.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from scipy import ndimage
from .classes import NUM_CLASSES,CLASS_NAMES
from .geometry import letterbox,native_prediction
from .model import model_from_checkpoint
from .io import choose_device
from .engine import precision


@dataclass
class Prediction:
    class_ids: np.ndarray
    disease_probability: np.ndarray
    confidence: np.ndarray
    decision: dict


def screen_prediction(class_ids, probability, threshold=0.5, uncertain_threshold=0.25,
                      min_area=8, mode='component', leaf_mask=None):
    """Illustrative, UNCALIBRATED routing policy; not a healthy-leaf diagnosis.

    mode='any': the user's literal any-nonbackground-pixel rule.
    mode='component': confirmed components or uncertain evidence trigger diffusion.
    Masks are not altered by routing thresholds.
    """
    if not 0 <= uncertain_threshold <= threshold <= 1 or min_area < 1:
        raise ValueError('Require 0<=uncertain_threshold<=threshold<=1 and min_area>=1')
    ids=np.asarray(class_ids); p=np.asarray(probability)
    if ids.shape!=p.shape: raise ValueError('Mask/probability shape mismatch')
    valid=np.ones(ids.shape,dtype=bool) if leaf_mask is None else np.asarray(leaf_mask,dtype=bool)
    if valid.shape!=ids.shape or not valid.any(): raise ValueError('Invalid or empty leaf mask')
    raw=(ids>0)&(ids<NUM_CLASSES)&valid
    evidence=raw&(p>=threshold)
    components,n=ndimage.label(evidence,structure=np.ones((3,3),dtype=np.uint8))
    areas=np.bincount(components.ravel())[1:]
    largest=int(areas.max()) if areas.size else 0
    if mode=='any':
        status='suspected_disease' if raw.any() else 'no_disease_detected'
    elif mode=='component':
        if largest>=min_area: status='suspected_disease'
        elif raw.any() or np.any(valid&(p>=uncertain_threshold)): status='uncertain'
        else: status='no_disease_detected'
    else: raise ValueError('mode must be any or component')
    return {'status':status,'needs_diffusion':status!='no_disease_detected',
            'any_nonbackground':bool(raw.any()),'predicted_disease_pixels':int(raw.sum()),
            'largest_confident_component':largest,'max_disease_probability':float(p[valid].max()),
            'threshold':float(threshold),'uncertain_threshold':float(uncertain_threshold),
            'min_component_pixels':int(min_area),'gate_mode':mode,
            'note':'Routing thresholds are provisional; calibrate on validation images. Zero prediction is not proof of health.'}


class PlantSegSegmenter:
    def __init__(self,checkpoint,device='cuda',image_size=None,amp=True):
        self.device=choose_device(device)
        self.model,self.checkpoint=model_from_checkpoint(checkpoint)
        if self.checkpoint['step']<1: raise ValueError('Checkpoint has no PlantSeg training updates')
        self.model.to(self.device).eval().requires_grad_(False)
        self.image_size=image_size or self.checkpoint['config']['data']['image_size']
        self.amp,self.amp_dtype=precision(self.device,amp)

    @torch.no_grad()
    def predict(self,image,threshold=0.5,uncertain_threshold=0.25,min_area=8,gate_mode=None,leaf_mask=None):
        if isinstance(image, (str, Path)):
            from .data import load_rgb
            image=load_rgb(image)
        elif isinstance(image, np.ndarray):
            if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
                raise ValueError('Array input must be H,W,3 RGB uint8; convert tensors/ranges explicitly.')
            image=Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise TypeError('Input must be a PIL RGB image, uint8 RGB array, or image path')
        image=image.convert('RGB')
        pixels,_,geom=letterbox(image,None,self.image_size)
        with torch.autocast(device_type=self.device.type,dtype=self.amp_dtype,enabled=self.amp):
            logits=self.model(pixel_values=pixels[None].to(self.device)).logits
        ids,p,confidence=native_prediction(logits,geom)
        if leaf_mask is not None:
            valid=np.asarray(leaf_mask,dtype=bool)
            if valid.shape!=ids.shape or not valid.any():
                raise ValueError('leaf_mask must match the input image grid and contain a leaf')
            ids[~valid]=0; p[~valid]=0; confidence[~valid]=0
        if gate_mode is not None:
            decision=screen_prediction(ids,p,threshold,uncertain_threshold,min_area,gate_mode,leaf_mask)
        else:
            disease=(ids>0)&(ids<NUM_CLASSES)
            decision={'status':'disease_predicted' if disease.any() else 'no_disease_predicted',
                'needs_diffusion':None, 'any_nonbackground':bool(disease.any()),
                'predicted_disease_pixels':int(disease.sum()), 'gate_mode':None,
                'note':'No routing/gating. Mask predicted from the supplied RGB. Background is not proof of health.'}
        return Prediction(ids,p,confidence,decision)


def class_summary(labels):
    values,counts=np.unique(labels,return_counts=True)
    return [{'id':int(i),'name':CLASS_NAMES[int(i)],'pixels':int(n)}
            for i,n in zip(values,counts) if 0<int(i)<NUM_CLASSES]
