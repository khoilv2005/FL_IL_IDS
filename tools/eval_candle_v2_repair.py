"""Evaluate the registered-router repair from existing CANDLE V2 results, no training."""
import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def extract_roles(path,destination):
    destination.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(path) as archive:
        manifests=[n for n in archive.namelist() if PurePosixPath(n).name=='role_manifest.json']
        if len(manifests)!=1:raise ValueError('Choose a roles ZIP containing one manifest')
        prefix=manifests[0][:-len('role_manifest.json')]
        for name in archive.namelist():
            if not name.startswith(prefix) or name.endswith('/'):continue
            relative=name[len(prefix):]
            parts=PurePosixPath(relative)
            if parts.is_absolute() or '..' in parts.parts or '\\' in relative:
                raise ValueError('Unsafe roles archive member')
            if relative not in ('role_manifest.json','role_lock.json') and not re.fullmatch(r'indices/client_\d+\.npz',relative):
                raise ValueError(f'Unexpected roles archive member: {relative}')
            target=destination.joinpath(*parts.parts)
            target.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(name) as source,target.open('wb') as output:
                shutil.copyfileobj(source,output,8*1024*1024)
    return destination


def prepare_inputs(results_root,work,checkpoint=None,roles=None):
    """Resolve mounted outputs or extract only task5 + roles from an outer ZIP."""
    root=Path(results_root)
    origin={}
    def result_parent(chosen):
        start=chosen if chosen.is_dir() else chosen.parent
        for parent in [start,*start.parents]:
            if (parent/'candle_v2_protocol.json').is_file() or (parent/'candle_v2_completion.json').is_file():
                return parent
            if parent==root:break
        return chosen.parent
    if checkpoint:
        chosen=Path(checkpoint)
        if not chosen.is_file() and not (chosen/'checkpoint_archive_manifest.json').is_file():raise FileNotFoundError(chosen)
        parent=result_parent(chosen);origin=dict(checkpoint=str(chosen),source_kind='explicit checkpoint')
    else:
        direct=[p for p in root.rglob('checkpoint_task_5_all_rounds.zip')
                if (p.parent/'candle_v2_protocol.json').is_file() or (p.parent/'candle_v2_completion.json').is_file()] if root.is_dir() else []
        if len(direct)>1:raise ValueError('Multiple CANDLE V2 runs found; set CHECKPOINT explicitly')
        if not direct and root.is_dir():
            for manifest in root.rglob('checkpoint_archive_manifest.json'):
                value=read_json(manifest)
                if int(value.get('task_id',-1))==5:
                    parent=result_parent(manifest.parent)
                    if (parent/'candle_v2_protocol.json').is_file() or (parent/'candle_v2_completion.json').is_file():
                        direct.append(manifest.parent)
            if len(direct)>1:raise ValueError('Multiple extracted CANDLE V2 runs found; set CHECKPOINT explicitly')
        if direct:
            chosen=direct[0];parent=result_parent(chosen)
            origin=dict(checkpoint=str(chosen),source_kind='mounted extracted results')
        else:
            candidates=[]
            archives=[root] if root.is_file() else sorted(root.rglob('*.zip'))
            for path in archives:
                try:
                    with zipfile.ZipFile(path) as archive:
                        names=archive.namelist()
                        for name in names:
                            if not name.endswith('candle_v2_protocol.json'):continue
                            prefix=name[:-len('candle_v2_protocol.json')]
                            if prefix+'checkpoint_task_5_all_rounds.zip' in names:
                                candidates.append((path,prefix))
                except zipfile.BadZipFile:
                    continue
            if len(candidates)!=1:
                raise ValueError(f'Found {len(candidates)} CANDLE V2 runs. Mount results (14), or set RESULTS_ROOT / CHECKPOINT to one run')
            path,prefix=candidates[0];parent=work/'mounted_run';parent.mkdir()
            # Whitelist avoids unpacking other five tasks or exporting checkpoint duplicates.
            selected=['checkpoint_task_5_all_rounds.zip','candle_v2_roles.zip',
                      'candle_v2_protocol.json','candle_v2_completion.json',
                      'candle_v2_self_task_5/confusion_matrix.json','candle_v2_self_task_5.zip']
            with zipfile.ZipFile(path) as archive:
                names=set(archive.namelist())
                for name in selected:
                    if prefix+name not in names:continue
                    target=parent/name;target.parent.mkdir(parents=True,exist_ok=True)
                    print('Preparing input:',name,flush=True)
                    with archive.open(prefix+name) as source,target.open('wb') as output:
                        shutil.copyfileobj(source,output,8*1024*1024)
            chosen=parent/'checkpoint_task_5_all_rounds.zip'
            origin=dict(outer_archive=str(path),prefix=prefix,checkpoint=str(chosen),source_kind='nested results ZIP')
    if roles:
        role_root=Path(roles)
        if not (role_root/'role_manifest.json').is_file():
            matches=[p.parent for p in role_root.rglob('role_manifest.json') if (p.parent/'role_lock.json').is_file()]
            if len(matches)!=1:raise ValueError('Set ROLES_DIR to one manifest directory')
            role_root=matches[0]
    elif (parent/'candle_v2_roles.zip').is_file():
        role_root=extract_roles(parent/'candle_v2_roles.zip',work/'roles')
    else:
        matches=[p.parent for p in parent.rglob('role_manifest.json') if (p.parent/'role_lock.json').is_file()]
        if len(matches)!=1:
            raise FileNotFoundError('Choose the exact original roles ZIP or extracted manifest directory via ROLES_DIR')
        role_root=matches[0]
    expected=None
    matrix=parent/'candle_v2_self_task_5/confusion_matrix.json'
    if matrix.is_file():expected=read_json(matrix)
    elif (parent/'candle_v2_self_task_5.zip').is_file():
        with zipfile.ZipFile(parent/'candle_v2_self_task_5.zip') as archive:
            matches=[n for n in archive.namelist() if PurePosixPath(n).name=='confusion_matrix.json']
            if len(matches)==1:expected=json.loads(archive.read(matches[0]))
    origin.update(roles=str(role_root),original_confusion_present=expected is not None)
    return chosen,role_root,expected,origin


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root',required=True)
    p.add_argument('--data-dir',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--roles')
    p.add_argument('--work-dir')
    p.add_argument('--batch-size',type=int,default=4096)
    p.add_argument('--device',default='cuda')
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    output=Path(a.output).resolve()
    if output.exists() or Path(str(output)+'.zip').exists():
        raise FileExistsError('Use a new output path; never overwrite original results or an archive')
    if a.batch_size<1:raise ValueError('Positive batch size required')
    temp_parent=Path(a.work_dir or ('/kaggle/temp' if Path('/kaggle').exists() else tempfile.gettempdir()))
    temp_parent.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='candle_v2_repair_',dir=temp_parent))
    checkpoint,roles,expected,origin=prepare_inputs(a.results_root,work,a.checkpoint,a.roles)
    if a.prepare_only:
        output.mkdir(parents=True);write(output/'prepared_inputs.json',origin)
        print(json.dumps(origin,indent=2));return
    import torch
    from tools.train_candle_v1 import resolve_data
    from tools.eval_candle_v2 import run
    from fed_learning.strategies.decentralized.candle_v2_inference import ROUTED
    if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('Enable Kaggle GPU')
    data=resolve_data(a.data_dir)
    print('CANDLE V2 repair: saved binary router + registered class_tasks; frozen weights.',flush=True)
    print('One local model, one backbone forward; paired task-free control; no training.',flush=True)
    result=run(checkpoint,roles,data,output,a.device,a.batch_size,ROUTED,expected_taskfree_confusion=expected)
    write(output/'repair_inputs.json',origin)
    write(output/'repair_comparison.json',dict(completed=True,policy=ROUTED,
        baseline=result['paired_taskfree_control'],repaired=result['metrics'],
        delta_accuracy_pp=100*(result['metrics']['accuracy']-result['paired_taskfree_control']['accuracy']),
        rescue=result['rescue'],breaks=result['breaks'],test_rows=result['test_rows'],
        route_accuracy=result['route_accuracy'],original_baseline_matches=result['original_taskfree_confusion_matches'],
        no_training=True,limitation='Evaluation after development-selected inference repair, not untouched scientific confirmation'))
    shutil.make_archive(str(output),'zip',root_dir=output)
    print(json.dumps(read_json(output/'repair_comparison.json'),indent=2),flush=True)


if __name__=='__main__':main()
