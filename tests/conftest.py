from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
import pytest
from common.prepare import prepare_rows
from plantseg_seg.predictor import PlantSegSegmenter
from plantseg_seg.classes import NUM_CLASSES

torch.set_num_threads(2)


def make_source(tmp_path,ntrain=3,sizes=None):
    root=tmp_path/'raw';root.mkdir(parents=True)
    splits=['train']*ntrain+['val','test','test'];rows=[]
    sizes=sizes or [(61,39),(99,67),(17,15),(57,41),(83,47),(3,2)]
    for i,split in enumerate(splits):
        w,h=sizes[i%len(sizes)];a=np.random.default_rng(i).integers(0,255,(h,w,3),dtype=np.uint8)
        mask=np.zeros((h,w),np.uint8);mask[h//4:max(h//4+1,3*h//4),w//4:max(w//4+1,3*w//4)]=98
        im=root/f'{i}.png';gt=root/f'{i}_mask.png';Image.fromarray(a).save(im);Image.fromarray(mask).save(gt)
        rows.append({'id':f'image_{i}','image':str(im),'mask':str(gt),'split':split})
    return rows


@pytest.fixture
def prepared(tmp_path):
    rows=make_source(tmp_path);out=prepare_rows(rows,tmp_path/'prepared',preview_count=1)
    return out,rows


class ToyConfig:
    plantseg_training_updates=0
    initializer_range=.02
    num_labels=116
    def to_dict(self):return {'num_labels':116,'model_type':'toy_NOT_A_TRAINED_SEGFORMER','plantseg_training_updates':self.plantseg_training_updates}


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.segformer=torch.nn.Conv2d(3,4,1);self.decode_head=torch.nn.Module()
        self.decode_head.classifier=torch.nn.Conv2d(4,116,1);self.config=ToyConfig()
    def forward(self,pixel_values):return SimpleNamespace(logits=self.decode_head.classifier(self.segformer(pixel_values)))


class ColorModel(torch.nn.Module):
    def forward(self,pixel_values):
        b,_,h,w=pixel_values.shape;logits=torch.zeros(b,116,max(1,h//4),max(1,w//4))
        idx=98 if pixel_values[:,0].mean()>pixel_values[:,2].mean() else 97
        logits[:,idx]=8;return SimpleNamespace(logits=logits)


def fake_segmenter(meta,canvas=32):
    s=PlantSegSegmenter.__new__(PlantSegSegmenter);s.device=torch.device('cpu');s.model=ColorModel().eval()
    s.image_size=canvas;s.amp=False;s.amp_dtype=torch.float16;s.checkpoint={'manifest_sha256':meta['manifest_sha256'],'step':3}
    return s


def training_config(prepared):
    return {'model':{'pretrained_dir':'NOT_USED_IN_TEST'},'data':{'prepared':str(prepared),'image_size':32,'workers':0,'augment':True,'degrade_scales':[1],'verify_content':True},
      'train':{'device':'cpu','seed':42,'batch_size':1,'accumulate':2,'epochs':1,'steps':None,'lr_encoder':1e-4,'lr_decoder':1e-3,'weight_decay':.01,
          'warmup':0,'grad_clip':1.,'amp':False,'trainable':'all','disease_dice_weight':0.,'save_every':1,'validate_every':0,'val_limit':0,'log_every':1,
          'patience':0,'selection_metric':'miou_foreground','preview_count':1,'save_val_predictions':True}}
