"""Exercise command wiring without substituting untrained toy models in user CLI."""
import sys
from pathlib import Path
import pytest
import run as router


@pytest.mark.parametrize('command',['prepare','audit','download-seg','check-seg','train-seg','download-sr','check-sr','train-sr','evaluate','infer-seg','infer','reconstruct','test'])
def test_subcommand_help(command):
    with pytest.raises(SystemExit) as e:router.build_parser().parse_args([command,'--help'])
    assert e.value.code==0


def test_sr_training_mode_translation(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(router,'launch',lambda py,args,label,copy_log_to=None:calls.append((py,args,label)))
    monkeypatch.setattr(router,'resolved',lambda *args,**kwargs:'fake_resolved.yaml')
    monkeypatch.setattr(router,'data_config',lambda a:{'prepared':'test_prepared'})
    monkeypatch.setattr(sys,'argv',['run.py','--sr-python','/sr/python','train-sr','--run',str(tmp_path/'run')])
    router.main();assert calls[0][0]=='/sr/python';args=calls[0][1];assert args[args.index('--mode')+1]=='disease'


def test_all_requires_finetuned_checkpoint_before_work(monkeypatch,tmp_path):
    calls=[];monkeypatch.setattr(router,'launch',lambda *args,**kwargs:calls.append(args))
    monkeypatch.setattr(sys,'argv',['run.py','evaluate','--checkpoint','fake.pth','--modes','all','--output',str(tmp_path/'out')])
    with pytest.raises(ValueError,match='sr-checkpoint'):router.main()
    assert not calls


def test_check_sr_uses_training_smoke_config_and_shared_prepared(monkeypatch):
    calls=[];resolved=[]
    monkeypatch.setattr(router,'launch',lambda py,args,label,copy_log_to=None:calls.append(args))
    monkeypatch.setattr(router,'resolved',lambda p,c,l:resolved.append((p,c,l)) or 'resolved.yaml')
    monkeypatch.setattr(router,'data_config',lambda a:{'prepared':'my_prepared'})
    monkeypatch.setattr(sys,'argv',['run.py','check-sr']);router.main()
    assert 'finetune' in resolved[0][0] and resolved[0][1]['data']['prepared']=='my_prepared'
