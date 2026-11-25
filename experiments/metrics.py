"""The two requested reconstruction metrics, on ORIGINAL unpadded RGB pixels."""
from __future__ import annotations
import math
import numpy as np
from skimage.metrics import structural_similarity


def rgb_metrics(prediction,reference):
    a=np.asarray(prediction,dtype=np.float64)/255.0;b=np.asarray(reference,dtype=np.float64)/255.0
    if a.shape!=b.shape or a.ndim!=3 or a.shape[2]!=3:raise ValueError('RGB targets must be aligned H,W,3')
    mse=float(np.mean((a-b)**2));psnr='Infinity' if mse==0 else float(-10*math.log10(mse))
    shortest=min(a.shape[:2]);win=min(7,shortest if shortest%2 else shortest-1)
    # Do not silently drop a tiny image from the shared cohort. SSIM has no standard
    # >=3-pixel window here; report null and the reason, but still score PSNR/masks.
    ssim=float(structural_similarity(a,b,channel_axis=-1,data_range=1.,win_size=win,
               gaussian_weights=False,use_sample_covariance=True)) if win>=3 else None
    return {'psnr_rgb_db':psnr,'ssim_rgb':ssim,'ssim_window':win if win>=3 else None,
            'ssim_note':None if win>=3 else 'SSIM undefined: shortest original side <3; sample retained for all other metrics'}


def aggregate_rgb(records):
    def agg(key):
        values=[r[key] for r in records if r.get(key) is not None]
        if not values:return None,None,0
        if 'Infinity' in values:return 'Infinity',None,len(values)
        x=np.asarray(values,dtype=float);return float(x.mean()),float(x.std(ddof=0)),len(values)
    p,ps,pn=agg('psnr_rgb_db');s,ss,sn=agg('ssim_rgb')
    return {'PSNR_dB':p,'SSIM':s,'psnr_std':ps,'ssim_std':ss,'psnr_images':pn,'ssim_images':sn}
