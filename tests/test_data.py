import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from common.prepare import prepare_rows,rows_from_config
from common.runtime import read_csv,read_json,write_csv
from plantseg_seg.audit import verify_prepared
from plantseg_seg.data import load_labels,PlantSegDataset
from plantseg_seg.classes import CLASS_NAMES,NUM_CLASSES
from experiments.cases import make_cases,verify_cases,pad_rgb
from leafdiff.data import PlantPairs,load_mask
from conftest import make_source


def test_class_mapping():
    assert NUM_CLASSES==116 and CLASS_NAMES[98]=='tomato early blight'
    assert CLASS_NAMES[115]=='zucchini yellow mosaic virus'


def test_exact_split_membership_preserved(prepared):
    root,rows=prepared;m,actual=verify_prepared(root)
    assert [(r['id'],r['split']) for r in rows]==[(r['id'],r['split']) for r in actual]
    assert m['counts']=={'train':3,'val':1,'test':2}


def test_manifest_tamper(prepared):
    root,_=prepared;rows=read_csv(root/'manifest.csv');rows[0]['split']='test';write_csv(root/'manifest.csv',rows)
    with pytest.raises(ValueError,match='manifest'):verify_prepared(root)


def test_mask_content_tamper(prepared):
    root,rows=prepared;p=rows[0]['mask'];a=np.array(Image.open(p));a[a==98]=97;Image.fromarray(a).save(p)
    with pytest.raises(ValueError,match='content'):verify_prepared(root)


def test_cross_split_duplicate_stops_without_resplitting(tmp_path):
    rows=make_source(tmp_path);Image.open(rows[0]['image']).save(rows[-2]['image']);Image.open(rows[0]['mask']).save(rows[-2]['mask'])
    with pytest.raises(ValueError,match='Cross-split'):prepare_rows(rows,tmp_path/'out')
    assert not (tmp_path/'out/manifest.csv').exists()


def test_same_split_duplicate_preserved(tmp_path):
    rows=make_source(tmp_path);Image.open(rows[0]['image']).save(rows[1]['image']);Image.open(rows[0]['mask']).save(rows[1]['mask'])
    root=prepare_rows(rows,tmp_path/'out',0);meta,actual=verify_prepared(root)
    assert len(actual)==len(rows) and meta['same_split_duplicates_preserved']==1


def test_size_mismatch_not_silently_skipped(tmp_path):
    rows=make_source(tmp_path);Image.fromarray(np.zeros((5,5),np.uint8)).save(rows[0]['mask'])
    with pytest.raises(ValueError,match='dimensions'):prepare_rows(rows,tmp_path/'out')


def test_palette_index_ids_preserved(tmp_path):
    a=np.array([[0,98],[115,255]],np.uint8);im=Image.fromarray(a).convert('P');im.putpalette([200,0,20]*256);p=tmp_path/'p.png';im.save(p)
    assert np.array_equal(load_labels(p),a)


@pytest.mark.parametrize('case',['rgb','bad_ids','all_ignore'])
def test_invalid_masks_stop(tmp_path,case):
    a={'rgb':np.full((4,4,3),[10,20,30],dtype=np.uint8),'bad_ids':np.full((4,4),190,dtype=np.uint8),'all_ignore':np.full((4,4),255,dtype=np.uint8)}[case]
    p=tmp_path/'mask.png';Image.fromarray(a).save(p)
    with pytest.raises(ValueError):load_labels(p)


def test_shared_all_sizes_no_edges_removed(prepared,tmp_path):
    root,rows=prepared;c=make_cases(root,tmp_path/'cases',preview_count=1);meta,cases=verify_cases(c)
    selected=[r for r in rows if r['split']=='test'];assert len(cases)==len(selected)
    for row,src in zip(cases,selected):
        assert np.array_equal(np.array(Image.open(c/row['reference'])),np.array(Image.open(src['image'])))
        assert np.array_equal(np.array(Image.open(c/row['mask'])),np.array(Image.open(src['mask'])))
        assert Image.open(c/row['bicubic']).size==Image.open(src['image']).size
        assert int(row['padded_width'])%4==0 and int(row['padded_height'])%4==0
    assert (int(cases[-1]['width']),int(cases[-1]['height']))==(3,2)


def test_comparison_cache_source_change_stops(prepared,tmp_path):
    root,rows=prepared;c=make_cases(root,tmp_path/'cases',preview_count=1)
    a=np.asarray(Image.open(rows[-1]['image'])).copy();a[0,0]=[0,0,0];Image.fromarray(a).save(rows[-1]['image'])
    with pytest.raises(ValueError,match='content'):make_cases(root,c)


def test_comparison_artifact_tamper(prepared,tmp_path):
    c=make_cases(prepared[0],tmp_path/'cases',preview_count=1);r=read_csv(c/'cases.csv')[0]
    Image.new('RGB',(10,10)).save(c/r['lr'])
    with pytest.raises(ValueError,match='artifact'):verify_cases(c)


def test_split_lists_do_not_shuffle_or_holdout(tmp_path):
    rows=make_source(tmp_path);raw=tmp_path/'raw';lists={}
    for s in ['train','val','test']:
        p=tmp_path/(s+'.txt');p.write_text('\n'.join(Path(r['image']).name for r in rows if r['split']==s));lists[s]=str(p)
    # Make exact-stem masks in a separate folder.
    masks=tmp_path/'masks';masks.mkdir()
    for r in rows:Image.open(r['mask']).save(masks/Path(r['image']).name)
    # Remove _mask files from images directory index by move, keeping only RGB.
    for r in rows:Path(r['mask']).unlink()
    actual=rows_from_config({'source':{'mode':'lists','images':str(raw),'masks':str(masks),'split_lists':lists}})
    assert [r['split'] for r in actual]==[r['split'] for r in rows]


def test_fixed_training_patch_accepts_small_sources(prepared):
    root,_=prepared;ds=PlantPairs(root,'train',training=True,augment=False)
    assert len(ds)==3
    for i in range(len(ds)):
        b=ds[i];assert b['hr'].shape==(3,256,256) and b['lr'].shape==(3,64,64)
        assert 0<b['valid'].sum()<256*256
        assert b['lesion'][b['valid']==0].sum()==0


def test_diffusion_valid_background_and_ignore(tmp_path):
    a=np.zeros((5,6),np.uint8);a[:2]=255;p=tmp_path/'m.png';Image.fromarray(a).save(p)
    lesion,valid=load_mask(p);assert lesion.sum()==0 and valid.sum()==18


def test_all_ignore_random_crop_recovered_using_valid_pixel(tmp_path):
    rows=make_source(tmp_path,sizes=[(600,600)]);a=np.full((600,600),255,np.uint8);a[2,2]=98
    Image.fromarray(a).save(rows[0]['mask']);root=prepare_rows(rows,tmp_path/'out',0)
    ds=PlantPairs(root,'train',training=False);item=next(ds[i] for i in range(len(ds)) if ds.samples[i][0]['id']==rows[0]['id'])
    assert item['valid'].sum()>0
