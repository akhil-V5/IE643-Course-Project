from __future__ import annotations
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .classes import NUM_CLASSES,CLASS_HASH,CLASS_NAMES
from .data import PlantSegDataset,load_rgb,load_labels
from .geometry import letterbox,native_prediction
from .io import (seed_all,choose_device,write_json,file_hash,atomic_torch_save,
                 rng_state,restore_rng,new_directory,safe_load)
from .model import initialize_pretrained,model_from_checkpoint,set_trainable,optimizer_groups
from .losses import segmentation_loss
from .metrics import confusion_matrix,metrics_from_confusion
from .visuals import save_panel,save_prediction,save_ground_truth
from .io import write_csv
from common.runtime import append_json
from .audit import verify_prepared


def precision(device,enabled=True):
    enabled=bool(enabled and device.type=='cuda')
    dtype=torch.bfloat16 if enabled and torch.cuda.is_bf16_supported() else torch.float16
    return enabled,dtype


def epoch_batches(n,batch_size,seed,epoch):
    generator=torch.Generator().manual_seed(seed+epoch)
    ids=torch.randperm(n,generator=generator).tolist()
    return [ids[i:i+batch_size] for i in range(0,n,batch_size)]


def make_train_iterator(dataset,cfg,batches_seen):
    batch_size=cfg['train']['batch_size']; n_batches=math.ceil(len(dataset)/batch_size)
    epoch,offset=divmod(batches_seen,n_batches)
    dataset.epoch=epoch
    batches=epoch_batches(len(dataset),batch_size,cfg['train']['seed'],epoch)[offset:]
    # Each sample augmentation is deterministic for (image, epoch); resumption is not
    # dependent on which worker loaded it. Persistent workers are intentionally off.
    workers=cfg['data']['workers']
    loader=DataLoader(dataset,batch_sampler=batches,num_workers=workers,
                      pin_memory=cfg['train']['device'].startswith('cuda'),
                      generator=torch.Generator().manual_seed(cfg['train']['seed']+epoch))
    return iter(loader),epoch


@torch.no_grad()
def validate(model,dataset,device,amp,amp_dtype,preview_dir=None,preview_count=12,save_predictions=True):
    model.eval(); cm=np.zeros((NUM_CLASSES,NUM_CLASSES),dtype=np.int64)
    total_loss=0.0;details=[]
    out=Path(preview_dir) if preview_dir else None
    for i,row in enumerate(dataset.rows):
        image=load_rgb(row['image']); target=load_labels(row['mask'])
        pixels,small_target,geom=letterbox(image,target,dataset.image_size)
        with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=amp):
            logits=model(pixel_values=pixels[None].to(device)).logits
        loss,_=segmentation_loss(logits,small_target[None].to(device),0)
        pred,prob,_=native_prediction(logits,geom)
        current=confusion_matrix(pred,target);cm+=current
        one,_=metrics_from_confusion(current)
        total_loss+=float(loss)
        details.append({'id':row['id'],'loss':float(loss),'miou_foreground':one['miou_foreground'],
                        'disease_dice':one['disease_dice']})
        if out:
            save_this_image = (preview_count == 0 or i < preview_count)

            if save_predictions and save_this_image:
                save_prediction(
                    image, pred, prob, out, row['id'], preview=False
                )
                save_ground_truth(
                    image, target, out, row['id']
                )

            if save_this_image:
                save_panel(
                    image, pred, target,
                    out / 'panels' / (row['id'] + '.png')
                )
        if (i+1)%25==0 or i+1==len(dataset):print(f'Validation {i+1}/{len(dataset)}',flush=True)
    metrics,classes=metrics_from_confusion(cm)
    metrics.update({'images':len(dataset),'cross_entropy_input_grid':total_loss/len(dataset),
                    'metric_grid':'Original annotation dimensions; predictions restored from letterbox.',
                    'monitor_subset':dataset.limit > 0})
    if out:
        out.mkdir(parents=True,exist_ok=True)
        write_csv(out/'per_image.csv',details);write_csv(out/'per_class.csv',classes)
        np.save(out/'confusion_matrix.npy',cm);write_json(out/'metrics.json',metrics)
    return metrics


def run_training(cfg,run_dir,resume=None,init=None,model_override=None):
    """model_override is only a unit-test injection point, not a CLI model option."""
    if resume and init: raise ValueError('Choose resume or init, not both')
    t=cfg['train']; seed_all(int(t['seed']))
    device=choose_device(t['device'])
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
    if t['accumulate']<1 or t['batch_size']<1: raise ValueError('Batch size/accumulation must be positive')
    if min(t['save_every'],t['log_every'])<1 or t['validate_every']<0:
        raise ValueError('Save/log intervals must be positive; validation interval may be 0 (epoch boundaries)')
    verified_meta, _ = verify_prepared(cfg['data']['prepared'], verify_content=cfg['data'].get('verify_content', True))
    train_ds=PlantSegDataset(cfg['data']['prepared'],'train',cfg['data']['image_size'],training=True,
        augment=cfg['data']['augment'],seed=t['seed'],degrade_scales=cfg['data']['degrade_scales'])
    val_ds=PlantSegDataset(cfg['data']['prepared'],'val',cfg['data']['image_size'],seed=t['seed'],limit=t['val_limit'])
    manifest_hash=file_hash(Path(cfg['data']['prepared'])/'manifest.csv')
    if train_ds.meta['class_hash']!=CLASS_HASH: raise ValueError('Class mapping changed')
    if train_ds.meta['manifest_sha256']!=manifest_hash: raise ValueError('Manifest changed after audit; create a fresh audited preparation')
    run=Path(run_dir).resolve()
    previous=None
    if resume:
        previous=safe_load(resume)
        if previous['manifest_sha256']!=manifest_hash: raise ValueError('Cannot resume on a different data split')
        for field in ('batch_size','accumulate','seed','trainable','lr_encoder','lr_decoder','weight_decay','disease_dice_weight','selection_metric'):
            if previous['config']['train'][field]!=t[field]:
                raise ValueError(f'Exact resume requires unchanged {field}; use --init for a new phase')
        for field in ('image_size','augment','degrade_scales'):
            if previous['config']['data'][field]!=cfg['data'][field]:
                raise ValueError(f'Resume requires unchanged {field}; use --init instead')
        run.mkdir(parents=True,exist_ok=True)
    else:
        new_directory(run)
    if model_override is not None:
        model=model_override; provenance={'source':'UNIT TEST MODEL, not a trained disease model'}
        if previous: model.load_state_dict(previous['model'])
    elif resume or init:
        model,loaded=model_from_checkpoint(resume or init)
        provenance=loaded['initialization']
    else:
        model,provenance=initialize_pretrained(cfg['model']['pretrained_dir'])
    set_trainable(model,t['trainable']); model.to(device)
    optimizer=torch.optim.AdamW(optimizer_groups(model,t['lr_encoder'],t['lr_decoder'],t['weight_decay']))
    amp,amp_dtype=precision(device,t['amp'])
    scaler=torch.amp.GradScaler('cuda',enabled=amp and amp_dtype==torch.float16)
    step,batches_seen,best=0,0,-float('inf')
    if previous:
        optimizer.load_state_dict(previous['optimizer'])
        scaler.load_state_dict(previous['scaler'])
        step,batches_seen,best=previous['step'],previous['batches_seen'],previous['best_score']
        restore_rng(previous['rng'])
    nbatches=math.ceil(len(train_ds)/t['batch_size'])
    if t.get('steps') is not None:
        target_steps=int(t['steps'])
        total_batches=batches_seen + max(0, target_steps-step)*t['accumulate']
    else:
        total_batches=int(t['epochs'])*nbatches
        remaining_batches = max(0, total_batches-batches_seen)
        target_steps=step + math.ceil(remaining_batches/t['accumulate'])
    if target_steps<=step: raise ValueError(f'Target steps={target_steps} must exceed saved step={step}')
    warmup=min(int(t['warmup']),max(0,target_steps//10))
    write_json(run/'config.json',cfg)
    write_json(run/'provenance.json',{'initialization':provenance,'manifest_sha256':manifest_hash,
       'class_hash':CLASS_HASH,'torch':str(torch.__version__),'device':str(device),
       'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
       'effective_batch':t['batch_size']*t['accumulate'],'target_steps':target_steps,
       'warmup_used':warmup,'resume':str(resume) if resume else None,'init':str(init) if init else None})
    print(f'Model=SegFormer; train images={len(train_ds)}; validation monitor={len(val_ds)}; '
          f'classes={NUM_CLASSES}; effective batch={t["batch_size"]*t["accumulate"]}; '
          f'target updates={target_steps}; whole-image letterbox={cfg["data"]["image_size"]}; warmup={warmup}',flush=True)
    iterator,epoch=make_train_iterator(train_ds,cfg,batches_seen)
    last_validation=-1
    bad_validations=int(previous.get('bad_validations', 0)) if previous else 0
    selection_metric=t.get('selection_metric', 'miou_foreground')
    if selection_metric not in {'miou_foreground', 'disease_dice'}:
        raise ValueError('selection_metric must be miou_foreground or disease_dice')
    def save(name):
        model.config.plantseg_training_updates=step
        atomic_torch_save({'task':'plantseg_multiclass_v1','class_hash':CLASS_HASH,
          'model':model.state_dict(),'model_config':model.config.to_dict(),'config':cfg,
          'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
          'step':step,'batches_seen':batches_seen,'best_score':best,'bad_validations':bad_validations,
          'selection_metric':selection_metric,'rng':rng_state(),
          'initialization':provenance,'manifest_sha256':manifest_hash},run/name)
    while step<target_steps:
        start=time.perf_counter(); model.train()
        previous_epoch=batches_seen//nbatches
        if t['trainable']=='decoder': model.segformer.eval()
        optimizer.zero_grad(set_to_none=True)
        next_step=step+1
        if warmup and next_step<=warmup:
            factor=next_step/warmup
        else:
            progress=(next_step-warmup)/max(1,target_steps-warmup)
            factor=0.1+0.9*(1+math.cos(math.pi*min(1,progress)))/2
        for group in optimizer.param_groups: group['lr']=group['base_lr']*factor
        n_accum=min(t['accumulate'],total_batches-batches_seen)
        if n_accum<=0: break
        logs=[]
        for _ in range(n_accum):
            try: batch=next(iterator)
            except StopIteration:
                iterator,epoch=make_train_iterator(train_ds,cfg,batches_seen)
                batch=next(iterator)
            x=batch['pixel_values'].to(device,non_blocking=True)
            y=batch['labels'].to(device,non_blocking=True)
            with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=amp):
                logits=model(pixel_values=x).logits
            loss,parts=segmentation_loss(logits,y,t['disease_dice_weight'])
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss; stop rather than save corrupt weights')
            scaler.scale(loss/n_accum).backward(); logs.append(parts); batches_seen+=1
        scaler.unscale_(optimizer)
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),t['grad_clip'])
        if not torch.isfinite(grad): raise FloatingPointError('Non-finite gradients; retry with --no-amp and inspect data')
        scaler.step(optimizer); scaler.update(); step=next_step
        if device.type=='cuda': torch.cuda.synchronize(device)
        elapsed=time.perf_counter()-start
        log={'step':step,'epoch_fraction':batches_seen/nbatches,'lr_max':max(g['lr'] for g in optimizer.param_groups),
             'grad_norm':float(grad),'seconds_per_update':elapsed,
             **{k:sum(x[k] for x in logs)/len(logs) for k in logs[0]}}
        with (run/'training.jsonl').open('a') as f: f.write(json.dumps(log)+'\n')
        if step==1 or step%t['log_every']==0: print(log,flush=True)
        epoch_finished=batches_seen//nbatches > previous_epoch
        do_validate = ((t['validate_every'] > 0 and step%t['validate_every']==0)
                       or (t['validate_every']==0 and epoch_finished) or step==target_steps)
        stop_early=False
        if do_validate:
            metrics=validate(model,val_ds,device,amp,amp_dtype,run/'validation'/f'step_{step:06d}',preview_count=t.get('preview_count',12),save_predictions=t.get('save_val_predictions',True))
            metrics['step']=step
            write_json(run/f'validation_{step:06d}.json',metrics)
            append_json(run/'validation.jsonl',metrics)
            score=metrics[selection_metric]
            if score is None: score=-metrics['cross_entropy_input_grid']
            if score>best:
                best=float(score); bad_validations=0; save('best.pth')
            else:
                bad_validations += 1
            patience=int(t.get('patience', 0))
            stop_early=patience>0 and bad_validations>=patience
            last_validation=step
            print('Validation:',{k:metrics[k] for k in ('step','miou_foreground','disease_iou','disease_dice','images')},flush=True)
        if step%t['save_every']==0 or step==target_steps or do_validate:
            save('last.pth')
        if stop_early:
            print(f'Early stop after {bad_validations} validation checks without improvement.',flush=True)
            break
    from .io import read_json
    import csv
    for name in ('training','validation'):
        jp=run/(name+'.jsonl')
        if jp.exists():
            records=[json.loads(line) for line in jp.read_text().splitlines() if line.strip()]
            write_csv(run/(name+'.csv'),records)
    write_json(run/'completed.json',{'steps':step,'batches_seen':batches_seen,'best_score':best,'selection_metric':selection_metric,'early_stopped':bool(stop_early)})
    print(f'Finished {step} updates. Checkpoints: {run}/best.pth and {run}/last.pth',flush=True)
    return run
