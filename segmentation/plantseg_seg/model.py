"""Thin adapter around the EXISTING Hugging Face SegFormer implementation.

NVIDIA ADE20K weights are reused, NOT claimed to be PlantSeg-trained weights.
Only decode_head.classifier is newly initialized for the PlantSeg classes.
"""
from __future__ import annotations
from pathlib import Path
import torch
from .classes import CLASS_NAMES, NUM_CLASSES, IGNORE_INDEX, CLASS_HASH
from .io import read_json, file_hash, safe_load


def hf_classes():
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except ImportError as e:
        raise ImportError('Install requirements.txt in the segmentation environment.') from e
    return SegformerConfig,SegformerForSemanticSegmentation


def adapt_classifier(model):
    old=model.decode_head.classifier
    classifier=torch.nn.Conv2d(old.in_channels,NUM_CLASSES,kernel_size=1)
    torch.nn.init.normal_(classifier.weight,std=model.config.initializer_range)
    torch.nn.init.zeros_(classifier.bias)
    model.decode_head.classifier=classifier.to(device=old.weight.device,dtype=old.weight.dtype)
    model.config.num_labels=NUM_CLASSES
    model.config.id2label=dict(enumerate(CLASS_NAMES))
    model.config.label2id={name:i for i,name in enumerate(CLASS_NAMES)}
    model.config.semantic_loss_ignore_index=IGNORE_INDEX
    model.config.plantseg_class_hash=CLASS_HASH
    model.config.plantseg_training_updates=0
    return model


def initialize_pretrained(folder):
    folder=Path(folder)
    Config,Model=hf_classes()
    config=Config.from_json_file(str(folder/'config.json'))
    if config.model_type!='segformer': raise ValueError('Expected SegFormer config')
    model=Model(config)
    st=folder/'model.safetensors'; pt=folder/'pytorch_model.bin'
    if st.exists():
        from safetensors.torch import load_file
        state=load_file(str(st),device='cpu'); weight_path=st
    elif pt.exists():
        # Only use the known official downloaded file. Never set weights_only=False.
        state=safe_load(pt); weight_path=pt
    else:
        raise FileNotFoundError(f'Run download_weights.py first: no weights in {folder}')
    lock=folder/'provenance.json'
    provenance=read_json(lock) if lock.exists() else {'source':'local pretrained weights'}
    if lock.exists():
        for filename, expected in provenance.get('sha256',{}).items():
            if not (folder/filename).is_file() or file_hash(folder/filename)!=expected:
                raise ValueError('Pretrained source file hash mismatch: '+filename)
    # Strict load before replacing classifier proves we did not silently skip backbone weights.
    model.load_state_dict(state,strict=True)
    del state
    model=adapt_classifier(model)
    provenance={**provenance,'adaptation':'Reused full ADE20K model except newly initialized 116-class classifier.',
                'class_hash':CLASS_HASH,'weight_file':weight_path.name}
    return model,provenance


def model_from_checkpoint(checkpoint):
    ckpt=safe_load(checkpoint)
    if ckpt.get('task')!='plantseg_multiclass_v1' or ckpt.get('class_hash')!=CLASS_HASH:
        raise ValueError('Expected a PlantSeg multiclass checkpoint produced by this package; not an ADE20K checkpoint')
    Config,Model=hf_classes()
    config=Config.from_dict(ckpt['model_config'])
    if config.num_labels!=NUM_CLASSES: raise ValueError('Checkpoint must have 116 classes')
    model=Model(config)
    model.load_state_dict(ckpt['model'],strict=True)
    return model,ckpt


def set_trainable(model,mode='all'):
    if mode not in {'all','decoder'}: raise ValueError('trainable must be all or decoder')
    for name,param in model.named_parameters():
        param.requires_grad_(mode=='all' or name.startswith('decode_head.'))


def optimizer_groups(model,encoder_lr,decoder_lr,weight_decay):
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        is_decoder=name.startswith('decode_head.')
        decay=0.0 if p.ndim==1 or name.endswith('.bias') else weight_decay
        lr=decoder_lr if is_decoder else encoder_lr
        key=(lr,decay)
        groups.setdefault(key,[]).append(p)
    return [{'params':params,'lr':lr,'base_lr':lr,'weight_decay':decay}
            for (lr,decay),params in groups.items()]
