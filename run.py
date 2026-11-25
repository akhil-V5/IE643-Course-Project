#!/usr/bin/env python
"""Project command router. Run --help or a subcommand --help.
Paths are relative to THIS suite root. Model environments execute in SEPARATE processes.
All commands save console logs, command lines, exit codes and resolved configuration.
"""
from __future__ import annotations
import argparse,json,os,shutil,subprocess,sys,time,uuid
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parent
MODES=['original','bicubic','resshift-pretrained','resshift-finetuned']


def read_yaml(path):
    with Path(path).open(encoding='utf-8') as f:return yaml.safe_load(f)


def write_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')


def launch(python,arguments,label,copy_log_to=None):
    stamp=time.strftime('%Y%m%d-%H%M%S')+'_'+uuid.uuid4().hex[:8]
    log=ROOT/'logs'/f'{stamp}_{label}.log';log.parent.mkdir(exist_ok=True)
    env=os.environ.copy();env['PYTHONUNBUFFERED']='1'
    env['PYTHONPATH']=os.pathsep.join([str(ROOT),str(ROOT/'segmentation'),str(ROOT/'magnification/resshift'),env.get('PYTHONPATH','')])
    cmd=[str(python),'-u',*map(str,arguments)];start=time.time()
    print('RUN:',subprocess.list2cmdline(cmd),'\nLOG:',log,flush=True)
    code=-1
    with log.open('w',encoding='utf-8') as f:
        f.write('COMMAND: '+subprocess.list2cmdline(cmd)+'\n');f.flush()
        try:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line,end='',flush=True);f.write(line);f.flush()
            code=proc.wait()
        except KeyboardInterrupt:
            if 'proc' in locals():proc.terminate();proc.wait()
            raise
        finally:
            f.write(f'\nEXIT CODE: {code}; ELAPSED SECONDS: {time.time()-start:.3f}\n')
    record={'argv':cmd,'cwd':str(ROOT),'exit_code':code,'seconds':time.time()-start,'log':str(log)}
    write_json(log.with_suffix('.json'),record)
    if copy_log_to and Path(copy_log_to).exists():
        shutil.copy2(log,Path(copy_log_to)/(label+'_console.log'));write_json(Path(copy_log_to)/(label+'_command.json'),record)
    if code:raise SystemExit(f'{label} failed (exit {code}); see {log}')
    return log


def data_config(a):return read_yaml(a.data_config)


def resolved(path,changes,label):
    cfg=read_yaml(path)
    for group,values in changes.items():cfg.setdefault(group,{}).update(values)
    p=ROOT/'logs/resolved_configs'/f'{label}_{uuid.uuid4().hex[:10]}.yaml';p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(yaml.safe_dump(cfg,sort_keys=False));return str(p)


def add_data(p):p.add_argument('--data-config',default='configs/data.yaml')
def add_ck(p):p.add_argument('--checkpoint',required=True,help='PlantSeg-finetuned SegFormer checkpoint')
def add_infer(p):
    add_ck(p);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--image-size',type=int);p.add_argument('--no-amp',action='store_true')
    p.add_argument('--preview-count',type=int,default=12);p.add_argument('--mask-dir')


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seg-python',default=sys.executable,help='Python executable in segmentation environment')
    p.add_argument('--sr-python',default=sys.executable,help='Python executable in separate ResShift environment')
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare',help='Audit existing splits; never create a new split');add_data(q)
    q=sub.add_parser('audit',help='Verify source content and audit resizing');add_data(q);q.add_argument('--image-size',type=int,default=512);q.add_argument('--output',default='data/resolution_audit.csv')
    q=sub.add_parser('download-seg');q.add_argument('--config',default='segmentation/configs/segformer_b2.yaml');q.add_argument('--revision',default='main')
    q=sub.add_parser('check-seg');q.add_argument('--config',default='segmentation/configs/segformer_b2.yaml');q.add_argument('--check-model',action='store_true');q.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    q=sub.add_parser('train-seg');add_data(q);q.add_argument('--config',default='segmentation/configs/segformer_b2.yaml');q.add_argument('--run',required=True);q.add_argument('--resume');q.add_argument('--init')
    g=q.add_mutually_exclusive_group();g.add_argument('--epochs',type=int);g.add_argument('--steps',type=int)
    for flag in ['batch-size','accumulate','workers','image-size','preview-count','val-limit','save-every','validate-every','warmup']:q.add_argument('--'+flag,type=int)
    q.add_argument('--trainable',choices=['all','decoder']);q.add_argument('--device',choices=['cuda','cpu']);q.add_argument('--no-amp',action='store_true')
    q=sub.add_parser('download-sr');q.add_argument('--ref',default='journal');q.add_argument('--skip-weights',action='store_true')
    q=sub.add_parser('check-sr');add_data(q);q.add_argument('--config',default='magnification/configs/resshift_finetune.yaml')
    q=sub.add_parser('train-sr',help='Optional future fine-tuning phase; same shared source splits');add_data(q);q.add_argument('--config',default='magnification/configs/resshift_finetune.yaml');q.add_argument('--run',required=True)
    q.add_argument('--resume');q.add_argument('--init');q.add_argument('--mode',choices=['disease-aware','baseline'],default='disease-aware')
    for flag in ['steps','batch-size','accumulate','workers','val-limit','save-every','validate-every']:q.add_argument('--'+flag,type=int)
    q.add_argument('--lr',type=float);q.add_argument('--aux-weight',type=float);q.add_argument('--lesion-multiplier',type=float);q.add_argument('--no-amp',action='store_true')
    q=sub.add_parser('evaluate',help='Shared-split end-to-end comparison; can select one or multiple modes');add_data(q);add_ck(q)
    q.add_argument('--config',default='configs/evaluation.yaml');q.add_argument('--split',choices=['val','test']);q.add_argument('--modes',nargs='+',choices=MODES+['all'])
    q.add_argument('--output',required=True);q.add_argument('--sr-config');q.add_argument('--sr-checkpoint');q.add_argument('--device');q.add_argument('--image-size',type=int)
    q.add_argument('--preview-count',type=int);q.add_argument('--limit',type=int);q.add_argument('--no-amp',action='store_true')
    q.add_argument('--variant',action='append',default=[],help='Use a previously generated mode=/absolute/path/manifest.json (verified, never rematched by basename)')
    q=sub.add_parser('infer-seg',help='Segment existing RGB files from ANY previous module');add_infer(q)
    q=sub.add_parser('infer',help='Raw LR -> chosen magnifier -> SegFormer, WITHOUT synthetic downsampling');add_infer(q)
    q.add_argument('--mode',choices=MODES,default='original');q.add_argument('--sr-config',default='magnification/configs/resshift_pretrained.yaml');q.add_argument('--sr-checkpoint')
    q=sub.add_parser('reconstruct',help='Magnification only; no segmentation model needed')
    q.add_argument('--mode',choices=['bicubic','resshift-pretrained','resshift-finetuned'],required=True);q.add_argument('--input',required=True);q.add_argument('--output',required=True)
    q.add_argument('--config',default='magnification/configs/resshift_pretrained.yaml');q.add_argument('--sr-checkpoint');q.add_argument('--device',default='cuda');q.add_argument('--no-amp',action='store_true');q.add_argument('--preview-count',type=int,default=12)
    q=sub.add_parser('test');q.add_argument('--optional-model',action='store_true',help='Include optional tiny-Transformers test when dependency is installed')
    return p


def pass_fields(a,names):
    out=[]
    for field in names:
        v=getattr(a,field,None)
        if v is not None:out.extend(['--'+field.replace('_','-'),str(v)])
    return out


def sr_stage(a,mode,output,comparison=None,input_path=None,preview_count=12,device='cuda',config=None):
    if mode=='resshift-finetuned' and not a.sr_checkpoint:raise ValueError('resshift-finetuned requires --sr-checkpoint; no fallback to pretrained')
    cfgpath='magnification/configs/bicubic.yaml' if mode=='bicubic' else (config or a.sr_config)
    updates={'preview_count':preview_count}
    if mode!='bicubic':updates.update(device=device,amp=not a.no_amp)
    cfgpath=resolved(cfgpath,{'inference':updates},mode)
    args=['-m','magnification.reconstruct','--config',cfgpath,'--output',output]
    args+=['--comparison',comparison] if comparison else ['--input',input_path]
    if mode=='resshift-finetuned':args+=['--checkpoint',a.sr_checkpoint]
    launch(a.seg_python if mode=='bicubic' else a.sr_python,args,mode,output)
    return str(Path(output)/'manifest.json')


def main():
    a=build_parser().parse_args();os.chdir(ROOT)
    if a.command=='prepare':launch(a.seg_python,['-m','common.prepare','--config',a.data_config],'prepare',data_config(a)['prepared'])
    elif a.command=='audit':launch(a.seg_python,['segmentation/audit_resolution.py','--prepared',data_config(a)['prepared'],'--image-size',a.image_size,'--output',a.output],'resolution_audit')
    elif a.command=='download-seg':launch(a.seg_python,['segmentation/download_weights.py','--config',a.config,'--revision',a.revision],'download_seg')
    elif a.command=='check-seg':launch(a.seg_python,['segmentation/check_env.py','--config',a.config,'--device',a.device]+(['--check-model'] if a.check_model else []),'check_seg')
    elif a.command=='train-seg':
        args=['segmentation/train_segmentation.py','--config',a.config,'--prepared',data_config(a)['prepared'],'--run',a.run]
        args+=pass_fields(a,['resume','init','epochs','steps','batch_size','accumulate','workers','image_size','preview_count','val_limit','save_every','validate_every','warmup','trainable','device'])
        if a.no_amp:args+=['--no-amp']
        launch(a.seg_python,args,'train_seg',a.run)
    elif a.command=='download-sr':launch(a.sr_python,['magnification/resshift/scripts/bootstrap.py','--ref',a.ref]+(['--skip-weights'] if a.skip_weights else []),'download_sr')
    elif a.command=='check-sr':
        cfg=resolved(a.config,{'data':{'prepared':data_config(a)['prepared']}},'check_sr')
        launch(a.sr_python,['magnification/resshift/check_backend.py','--config',cfg],'check_sr','runs/backend_check')
    elif a.command=='train-sr':
        cfg=resolved(a.config,{'data':{'prepared':data_config(a)['prepared']}},'train_sr')
        args=['magnification/resshift/train_diffusion.py','--config',cfg,'--run',a.run,'--mode',('disease' if a.mode=='disease-aware' else a.mode)]
        args+=pass_fields(a,['resume','init','steps','batch_size','accumulate','workers','val_limit','save_every','validate_every','lr','aux_weight','lesion_multiplier'])
        if a.no_amp:args+=['--no-amp']
        launch(a.sr_python,args,'train_sr',a.run)
    elif a.command=='evaluate':
        c=read_yaml(a.config);split=a.split or c['split'];limit=c['limit'] if a.limit is None else a.limit
        preview=c['preview_count'] if a.preview_count is None else a.preview_count
        modes=a.modes or c['modes'];modes=MODES if modes==['all'] else modes
        if 'all' in modes or len(set(modes))!=len(modes):raise ValueError('Use --modes all alone, or unique explicit modes')
        if min(limit,preview)<0:raise ValueError('Counts must be nonnegative')
        device=a.device or c['device'];canvas=a.image_size or c['image_size'];a.no_amp=a.no_amp or not c['amp']
        if 'resshift-finetuned' in modes and not a.sr_checkpoint and not any(x.startswith('resshift-finetuned=') for x in a.variant):raise ValueError('Supply --sr-checkpoint for the fine-tuned condition')
        out=Path(a.output).resolve()
        if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh evaluation output. Cached comparison cohort is reused separately.')
        cache=Path(c['cache_root'])/(split+'_whole_x4'+(f'_first{limit}' if limit else '_all'))
        prepared=data_config(a)['prepared']
        launch(a.seg_python,['-m','experiments.cases','--prepared',prepared,'--output',cache,'--split',split,'--limit',limit,'--preview-count',preview],'cases',cache)
        variants=dict(v.split('=',1) for v in a.variant)
        for mode in modes:
            if mode.startswith('resshift') and mode not in variants:
                variants[mode]=sr_stage(a,mode,str(out/'magnification'/mode),comparison=str(cache),preview_count=preview,device=device,config=a.sr_config or c['resshift_config'])
        args=['-m','experiments.evaluate','--checkpoint',a.checkpoint,'--comparison',cache,'--output',out/'evaluation','--modes',*modes,'--device',device,'--preview-count',preview]
        if canvas:args+=['--image-size',canvas]
        if a.no_amp:args+=['--no-amp']
        for mode,path in variants.items():args+=['--variant',mode+'='+path]
        launch(a.seg_python,args,'evaluate',out/'evaluation')
        for name in ['summary.csv','summary.json']:shutil.copy2(out/'evaluation'/name,out/name)
        print('RESULTS:',out/'summary.csv',flush=True)
    elif a.command in ('infer-seg','infer'):
        mode='original' if a.command=='infer-seg' else a.mode
        args=['-m','segmentation.infer','--checkpoint',a.checkpoint,'--device',a.device,'--preview-count',a.preview_count]
        if mode=='original':args+=['--input',a.input];dest=a.output
        else:
            if a.mask_dir:raise ValueError('For raw LR magnification GT must be supplied on output grid with exact IDs; use shared evaluate instead of ambiguous mask-dir matching')
            manifest=sr_stage(a,mode,str(Path(a.output)/'magnification'),input_path=a.input,preview_count=a.preview_count,device=a.device)
            args+=['--reconstruction-manifest',manifest];dest=str(Path(a.output)/'segmentation')
        args+=['--output',dest]
        if a.image_size:args+=['--image-size',a.image_size]
        if a.mask_dir:args+=['--mask-dir',a.mask_dir]
        if a.no_amp:args+=['--no-amp']
        launch(a.seg_python,args,'infer_seg',dest)
    elif a.command=='reconstruct':
        a.sr_config=a.config
        sr_stage(a,a.mode,a.output,input_path=a.input,preview_count=a.preview_count,device=a.device)
    elif a.command=='test':launch(a.seg_python,['-m','pytest','-q','tests'],'tests')


if __name__=='__main__':main()
