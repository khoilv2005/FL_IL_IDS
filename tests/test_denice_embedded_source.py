"""The upload contains exact current production source, not just a stale Git ref."""
import ast
import io
from pathlib import Path
import tempfile
import zipfile
from unittest.mock import patch
import json
import os
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('with_file', [True, False])
def test_single_upload_embeds_current_source(tmp_path, with_file):
    source = ROOT/'train_incremental_kaggle.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    helpers = [n for n in tree.body if isinstance(n,ast.FunctionDef) and
               n.name in ('embedded_cluster_source','_denice_support_payload')]
    scope = dict(Path=Path,tempfile=tempfile)
    if with_file:scope['__file__']=str(source)
    exec(compile(ast.Module(body=helpers,type_ignores=[]),str(source),'exec'),scope)
    with patch.object(tempfile,'mkdtemp',return_value=str(tmp_path)):
        bundle=scope['embedded_cluster_source']()
    with zipfile.ZipFile(bundle) as archive:
        expected=[*sorted((ROOT/'fed_learning').rglob('*.py')),
                  ROOT/'tools/train_denice_v2.py',ROOT/'tools/train_cluster_server.py']
        for p in expected:
            assert archive.read(p.relative_to(ROOT).as_posix())==p.read_bytes()
    assert source.read_bytes()==(ROOT/'output/single_file/train_incremental_kaggle.py').read_bytes()


def test_cell_bootstrap_without_file_or_history(monkeypatch,tmp_path):
    """Execute the actual first cell through fresh bootstrap, stopping at trainer."""
    from types import ModuleType
    for k in list(os.environ):
        if k.startswith('DENICE_'):monkeypatch.delenv(k)
    monkeypatch.setenv('DENICE_RESUME_DRIVE_URL','')
    monkeypatch.chdir(tmp_path)
    module=ModuleType('tools.train_cluster_server')
    calls=[]
    module.main=lambda:calls.append(os.environ['DENICE_SOURCE_ZIP'])
    monkeypatch.setitem(sys.modules,'tools.train_cluster_server',module)
    monkeypatch.setattr(sys,'path',list(sys.path))
    source=(ROOT/'train_incremental_kaggle.py').read_text(encoding='utf-8')
    namespace={'__name__':'__main__'}  # no __file__, IPython history or source on disk
    with pytest.raises(SystemExit) as done:
        exec(compile(source,'<kaggle-cell>','exec'),namespace)
    assert done.value.code==0 and len(calls)==1
    with zipfile.ZipFile(calls[0]) as archive:
        assert archive.read('tools/train_denice_v2.py')==(ROOT/'tools/train_denice_v2.py').read_bytes()


def test_env_adapter_forwards_compact_and_budget(monkeypatch,tmp_path):
    import importlib.util
    import sys
    from types import SimpleNamespace
    for k in list(__import__('os').environ):
        if k.startswith('DENICE_'):monkeypatch.delenv(k)
    (tmp_path/'metadata.json').write_text('{}')
    for k,v in dict(DENICE_DATA_DIR=str(tmp_path),DENICE_SERVER_EXCHANGE='1',
                    DENICE_COMPACT_UPLOADS='1',DENICE_PREDICTION_MODE='routerless_v1',
                    DENICE_SESSION_SECONDS='36000').items():monkeypatch.setenv(k,v)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False,device_count=lambda:0)))
    spec=importlib.util.spec_from_file_location('cluster_adapter',ROOT/'tools/train_cluster_server.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    calls=[]
    monkeypatch.setattr(module.subprocess,'run',lambda args,**kwargs:calls.append(args))
    module.main()
    args=calls[0]
    assert '--compact-uploads' in args and '--server-exchange' in args
    assert args[args.index('--session-seconds')+1]=='36000'


def test_resume_bootstrap_uses_saved_source_and_ignores_default_drive(monkeypatch,tmp_path):
    for k in list(os.environ):
        if k.startswith('DENICE_'):monkeypatch.delenv(k)
    saved=tmp_path/'saved';saved.mkdir()
    config=dict(server_only_storage=True,defer_evaluation=True,server_exchange=True,
        compact_uploads=True,policy='routerless_v1',batch_size=8,device='cpu',
        output_budget_gib=18,seed=42,worker_devices=[])
    (saved/'protocol_lock.json').write_text(json.dumps(dict(config=config)))
    (saved/'execution_provenance.json').write_text(json.dumps(dict(torch='test-runtime',commit='old')))
    with zipfile.ZipFile(saved/'checkpoint_task_0_round_002.zip','w') as z:
        z.writestr('rounds/002.json',json.dumps(dict(schema=1,task=0,round=2)))
    with zipfile.ZipFile(saved/'denice_cluster_server_source.zip','w') as z:
        z.writestr('tools/train_cluster_server.py','# original source sentinel')
    monkeypatch.setenv('DENICE_CLUSTER_RESUME_ROOT',str(saved))
    tree=ast.parse((ROOT/'train_incremental_kaggle.py').read_text(encoding='utf-8'))
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='configure_cluster_resume']
    branch=next(n for n in tree.body if isinstance(n,ast.If) and
                "TRAIN_PROFILE == 'cluster_server' and __name__ == '__main__'"==ast.unparse(n.test))
    calls=[]
    monkeypatch.setattr(subprocess,'run',lambda args,**kw:calls.append(args))
    extracted=tmp_path/'extracted';extracted.mkdir()
    monkeypatch.setattr(tempfile,'mkdtemp',lambda **kw:str(extracted))
    scope=dict(Path=Path,json=json,os=os,sys=sys,subprocess=subprocess,tempfile=tempfile,zipfile=zipfile,
        TRAIN_PROFILE='cluster_server',__name__='__main__',__file__=str(tmp_path/'upload.py'),
        CLUSTER_RESUME_DRIVE_URL='must-not-download',ensure_resume_torch=lambda _:None)
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'bootstrap','exec'),scope)
    with pytest.raises(SystemExit) as done:
        exec(compile(ast.Module(body=[branch],type_ignores=[]),'bootstrap','exec'),scope)
    assert done.value.code==0
    assert len(calls)==1
    assert Path(calls[0][-1]).read_text()=='# original source sentinel'
    assert os.environ['DENICE_COMPACT_UPLOADS']=='1'
    assert os.environ['DENICE_CLUSTER_RESUME_ROUND']=='2'
