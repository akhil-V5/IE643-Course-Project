from types import SimpleNamespace
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from common.runtime import read_json
from plantseg_seg.geometry import letterbox,native_prediction
from plantseg_seg.losses import segmentation_loss
from plantseg_seg.metrics import confusion_matrix,metrics_from_confusion
from plantseg_seg.model import adapt_classifier,set_trainable
from plantseg_seg.visuals import disease_gray,save_panel,save_prediction,save_ground_truth
from plantseg_seg.engine import run_training
from plantseg_seg.io import safe_load
from experiments.metrics import rgb_metrics,aggregate_rgb
from conftest import fake_segmenter,ToyModel,training_config


def test_logits_restored_before_argmax_matches_dense():
    _,_,g=letterbox(Image.new('RGB',(29,17)),None,32);logits=torch.randn(1,7,8,8)
    lab,p,conf=native_prediction(logits,g,chunk_size=2)
    dense=F.interpolate(logits,(32,32),mode='bilinear',align_corners=False)
    dense=dense[:,:,g.top:g.top+g.resized_height,g.left:g.left+g.resized_width]
    dense=F.interpolate(dense,(17,29),mode='bilinear',align_corners=False)[0]
    probs=dense.softmax(0)
    assert np.array_equal(lab,dense.argmax(0).numpy())
    assert np.allclose(p,(1-probs[0]).numpy(),atol=1e-6)
    assert np.allclose(conf,probs.max(0).values.numpy(),atol=1e-6)


@pytest.mark.parametrize('canvas',[32,64,96])
def test_changed_input_size_roundtrips(canvas):
    s=fake_segmenter({'manifest_sha256':'fake'},canvas)
    p=s.predict(Image.new('RGB',(83,47),(255,0,0)))
    assert p.class_ids.shape==(47,83) and p.decision['gate_mode'] is None


def test_prediction_uses_actual_new_rgb():
    s=fake_segmenter({'manifest_sha256':'fake'})
    a=s.predict(Image.new('RGB',(19,31),(255,0,0)));b=s.predict(Image.new('RGB',(19,31),(0,0,255)))
    assert (a.class_ids==98).all() and (b.class_ids==97).all()


def test_leaf_mask_clips_all_maps():
    s=fake_segmenter({'manifest_sha256':'fake'});m=np.zeros((31,19),bool);m[2:7,2:7]=True
    a=s.predict(Image.new('RGB',(19,31),(255,0,0)),leaf_mask=m)
    assert a.class_ids[~m].sum()==0 and a.disease_probability[~m].sum()==0


def test_wrong_disease_category_is_not_binary_localization_error():
    gt=np.array([[0,98],[98,0]],np.uint8);pred=np.array([[0,97],[97,0]],np.uint8)
    m,_=metrics_from_confusion(confusion_matrix(pred,gt))
    assert m['miou_foreground']==0 and m['disease_dice']==1


def test_false_positive_on_background_is_counted():
    gt=np.array([[98,0],[0,255]],np.uint8);pred=np.array([[98,98],[0,98]],np.uint8)
    m,_=metrics_from_confusion(confusion_matrix(pred,gt))
    assert m['miou_foreground']==.5 and m['disease_dice']==pytest.approx(2/3)
    assert m['valid_pixels']==3


def test_all_background_dice_is_undefined_not_perfect():
    x=np.zeros((3,3),np.uint8);m,_=metrics_from_confusion(confusion_matrix(x,x));assert m['disease_dice'] is None


def test_perfect_reconstruction_psnr_not_arbitrary_cap():
    x=np.random.default_rng(3).integers(0,255,(13,15,3),dtype=np.uint8);m=rgb_metrics(x,x)
    assert m['psnr_rgb_db']=='Infinity' and m['ssim_rgb']==pytest.approx(1)


def test_tiny_ssim_not_silently_excluded_from_other_metrics():
    x=np.ones((2,3,3),np.uint8);y=x*2;m=rgb_metrics(x,y)
    assert m['ssim_rgb'] is None and isinstance(m['psnr_rgb_db'],float)
    assert aggregate_rgb([m])['ssim_images']==0


def test_mask_outputs_visible_and_ids_unmodified(tmp_path):
    im=Image.new('RGB',(10,8),(45,65,75));gt=np.zeros((8,10),np.uint8);gt[2:5,3:7]=98;gt[0]=255
    pred=np.where(gt==255,0,gt).astype(np.uint8)
    save_ground_truth(im,gt,tmp_path,'sample');save_prediction(im,pred,None,tmp_path,'sample')
    assert set(np.unique(np.array(Image.open(tmp_path/'gt_disease_binary/sample.png'))))=={0,127,255}
    assert set(np.unique(np.array(Image.open(tmp_path/'disease_binary/sample.png'))))=={0,255}
    assert np.array_equal(np.array(Image.open(tmp_path/'class_ids/sample.png')),pred)
    save_panel(im,pred,gt,tmp_path/'panel.png');assert Image.open(tmp_path/'panel.png').width>=640


def test_ce_ignore_and_gradient():
    logits=torch.randn(1,116,8,8,requires_grad=True);y=torch.zeros(1,32,32,dtype=torch.long);y[:,5:12,5:12]=98;y[:,:2]=255
    loss,_=segmentation_loss(logits,y);loss.backward();assert torch.isfinite(loss) and logits.grad.abs().sum()>0


def test_adaptation_random_head_but_backbone_reused():
    m=ToyModel();before=m.segformer.weight.detach().clone();adapt_classifier(m)
    assert m.decode_head.classifier.out_channels==116 and torch.equal(before,m.segformer.weight)
    set_trainable(m,'decoder');assert not m.segformer.weight.requires_grad and m.decode_head.classifier.weight.requires_grad


def test_training_validation_resume_full_output(prepared,tmp_path):
    out,_=prepared;cfg=training_config(out);run=run_training(cfg,tmp_path/'run',model_override=ToyModel())
    c=safe_load(run/'last.pth');assert c['batches_seen']==3 and c['step']==2
    assert (run/'training.csv').is_file() and (run/'validation.csv').is_file()
    assert list((run/'validation/step_000002/gt_disease_binary').glob('*.png'))
    assert list((run/'validation/step_000002/disease_binary').glob('*.png'))
    cfg['train']['epochs']=2
    run_training(cfg,run,resume=run/'last.pth',model_override=ToyModel())
    c=safe_load(run/'last.pth');assert c['batches_seen']==6 and c['step']==4


def test_actual_transformers_api_optional():
    pytest.importorskip('transformers')
    from transformers import SegformerConfig,SegformerForSemanticSegmentation
    c=SegformerConfig(num_channels=3,num_labels=116,hidden_sizes=[8,16,32,64],depths=[1,1,1,1],
        num_attention_heads=[1,2,4,8],sr_ratios=[8,4,2,1],decoder_hidden_size=16)
    m=SegformerForSemanticSegmentation(c).eval();x=torch.randn(1,3,64,96);out=m(pixel_values=x).logits
    assert out.shape[1]==116;loss,_=segmentation_loss(out,torch.zeros(1,64,96,dtype=torch.long));loss.backward()
    assert m.decode_head.classifier.weight.grad is not None


def test_white_binary_preview_keeps_visible_boundary(tmp_path):
    from plantseg_seg.visuals import panel_grid
    white=Image.new('L',(83,47),255)
    path=tmp_path/'white_mask.png'
    panel_grid([white],['PREDICTED disease mask'],path,max_side=320)
    arr=np.asarray(Image.open(path))
    assert np.array_equal(arr[48,0],np.array([100,100,100]))
    assert np.array_equal(arr[49,1],np.array([255,255,255]))
