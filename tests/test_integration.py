import json,importlib.util,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import pytest
from PIL import Image
from common.runtime import read_json,read_csv,write_json,write_csv,digest
from experiments.cases import make_cases,verify_cases
from experiments.evaluate import evaluate,load_variant
from magnification.reconstruct import reconstruct
from segmentation.infer import infer
from conftest import fake_segmenter


class FakeSR:
    device=torch.device('cpu')
    def sample(self,x,amp=True):return F.interpolate(x,scale_factor=4,mode='nearest')


def sr_config(tmp_path):
    f=tmp_path/'NOT_REAL_WEIGHTS.pth';f.write_text('This is a synthetic test placeholder, not model weights.')
    return {'backend':{'diffusion_weights':str(f)},'inference':{'method':'resshift','scale':4,'tile_lr':64,'overlap':16,'seed':42,'device':'cpu','amp':False,'preview_count':1}}


def test_tiled_whole_image_odd_dimensions(tmp_path):
    file=Path(__file__).resolve().parents[1]/'magnification/resshift/infer.py'
    spec=importlib.util.spec_from_file_location('test_sr_infer',file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for w,h in [(3,2),(83,47),(129,137)]:
        arr=np.random.default_rng(3).integers(0,255,(h,w,3),dtype=np.uint8)
        pred, sampling = m.whole_image_reconstruct(
            FakeSR(), Image.fromarray(arr), amp=False
        )
        assert pred.size == (4*w, 4*h)
        assert sampling["sample_calls"] == 1
        expected=np.repeat(np.repeat(arr,4,0),4,1)
        assert np.max(np.abs(np.asarray(pred).astype(int)-expected))<=1


def test_full_comparison_every_source_all_modes(prepared,tmp_path):
    root,src=prepared;c=make_cases(root,tmp_path/'cases',preview_count=1)
    cfg=sr_config(tmp_path)
    sr=reconstruct(cfg,tmp_path/'sr',comparison=c,backend_override=FakeSR())
    ck=tmp_path/'seg_test_marker.bin';ck.write_text('NOT A SEGFORMER: test marker')
    meta=read_json(root/'metadata.json');s=fake_segmenter(meta)
    summary=evaluate(ck,c,tmp_path/'results',['original','bicubic','resshift-pretrained'],
        {'resshift-pretrained':str(sr/'manifest.json')},device='cpu',preview_count=1,segmenter_override=s)
    assert len(summary)==3 and {r['images'] for r in summary}=={2}
    assert summary[0]['PSNR_dB'] is None
    for mode in ['original','bicubic','resshift-pretrained']:
        assert len(list((tmp_path/'results'/mode/'disease_binary').glob('*.png')))==2
        assert len(list((tmp_path/'results'/mode/'gt_disease_binary').glob('*.png')))==2
    saved=read_json(tmp_path/'results/summary.json');assert not saved['actual_segformer_used']
    assert not saved['variant_provenance']['resshift-pretrained']['actual_backend']
    assert list((tmp_path/'results/comparison_panels').glob('*.png'))


def test_bicubic_reconstruct_matches_case_baseline(prepared,tmp_path):
    root,_=prepared;c=make_cases(root,tmp_path/'cases',preview_count=1)
    cfg={'inference':{'method':'bicubic','scale':4,'seed':42,'preview_count':1}}
    out=reconstruct(cfg,tmp_path/'bic',comparison=c)
    m,rows=verify_cases(c);info,mapping=load_variant(out/'manifest.json',m,rows)
    for r in rows:assert np.array_equal(np.array(Image.open(mapping[r['id']])),np.array(Image.open(c/r['bicubic'])))


def test_partial_reconstructions_stop(prepared,tmp_path):
    c=make_cases(prepared[0],tmp_path/'cases',preview_count=1);cfg=sr_config(tmp_path)
    sr=reconstruct(cfg,tmp_path/'sr',comparison=c,backend_override=FakeSR());rows=read_csv(sr/'images.csv');Path(sr/rows[0]['rgb']).unlink()
    meta,cases=verify_cases(c)
    with pytest.raises(ValueError,match='artifact'):load_variant(sr/'manifest.json',meta,cases)


def test_different_seg_training_manifest_stops(prepared,tmp_path):
    c=make_cases(prepared[0],tmp_path/'cases',preview_count=1);ck=tmp_path/'ck';ck.write_text('test')
    with pytest.raises(ValueError,match='DIFFERENT'):evaluate(ck,c,tmp_path/'eval',['original'],segmenter_override=fake_segmenter({'manifest_sha256':'other'}))


def test_changed_reconstruction_cohort_stops(prepared,tmp_path):
    c=make_cases(prepared[0],tmp_path/'cases',preview_count=1);sr=reconstruct(sr_config(tmp_path),tmp_path/'sr',comparison=c,backend_override=FakeSR())
    m,rows=verify_cases(c);m['cases_sha256']='other'
    with pytest.raises(ValueError,match='DIFFERENT'):load_variant(sr/'manifest.json',m,rows)


def test_finetuned_label_cannot_be_assigned_to_pretrained(prepared,tmp_path):
    c=make_cases(prepared[0],tmp_path/'cases',preview_count=1);sr=reconstruct(sr_config(tmp_path),tmp_path/'sr',comparison=c,backend_override=FakeSR())
    ck=tmp_path/'ck';ck.write_text('test')
    with pytest.raises(ValueError,match='mislabeled'):
        evaluate(ck,c,tmp_path/'eval',['resshift-finetuned'],{'resshift-finetuned':str(sr/'manifest.json')},segmenter_override=fake_segmenter(read_json(prepared[0]/'metadata.json')))


def test_standalone_inference_raw_and_reconstructed(tmp_path):
    im=tmp_path/'input.png';Image.new('RGB',(31,17),(180,30,20)).save(im);ck=tmp_path/'ck';ck.write_text('test')
    s=fake_segmenter({'manifest_sha256':'unused'})
    out=infer(ck,tmp_path/'pred',input_path=im,device='cpu',segmenter_override=s)
    assert read_json(out/'inference.json')['metrics'] is None
    cfg={'inference':{'method':'bicubic','scale':4,'seed':42,'preview_count':1}}
    rgb=reconstruct(cfg,tmp_path/'mag',input_path=im)
    pred=infer(ck,tmp_path/'after',reconstruction_manifest=rgb/'manifest.json',device='cpu',segmenter_override=s)
    mask=Image.open(next((pred/'class_ids').glob('*.png')));assert mask.size==(124,68)


def test_reject_output_inside_input(tmp_path):
    inp=tmp_path/'input';inp.mkdir();Image.new('RGB',(5,7)).save(inp/'x.png')
    cfg={'inference':{'method':'bicubic','scale':4,'seed':42}}
    with pytest.raises(ValueError,match='outside'):reconstruct(cfg,inp/'output',input_path=inp)
