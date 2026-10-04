"""Generate P1 notebook, reusing the original frozen panel preparation."""
import json
from pathlib import Path
from build_denice_tip_notebook import cell

ROOT = Path(__file__).resolve().parents[1]


def main():
    prior = json.loads((ROOT/'eval_denice_tip_router_kaggle.ipynb').read_text(encoding='utf-8'))
    setup = ''.join(prior['cells'][1]['source']).replace('denice_tip_03b9b53','denice_matched_03b9b53').replace('denice_tip_source','denice_matched_source')
    locate = '''# Attach the previous denice_tip_diagnostics.zip as a Kaggle dataset.
# Kaggle may expose it as an extracted directory; both formats are accepted.
DIAGNOSTIC_INPUT = ""
if not DIAGNOSTIC_INPUT:
    candidates = list(Path('/kaggle/input').rglob('denice_tip_diagnostics.zip'))
    candidates += [p.parent for p in Path('/kaggle/input').rglob('router_diagnostics.json')
                   if (p.parent/'predictions.csv').exists() and (p.parent/'profile_manifest.json').exists()]
    if len(candidates) != 1:
        raise FileNotFoundError(f'Attach the prior TIP diagnostic output or set DIAGNOSTIC_INPUT. Candidates: {candidates}')
    DIAGNOSTIC_INPUT = str(candidates[0])
print('Prior frozen predictions:',DIAGNOSTIC_INPUT)
'''
    download = ''.join(prior['cells'][2]['source'])
    prepare = ''.join(prior['cells'][3]['source'])
    prepare = prepare[:prepare.index('protocol = dict(')] + '''protocol = dict(
    kind='matched oracle and best allowed route; no router fitting',
    training_commit=config['git_commit'],evaluation_commit=source_commit,
    checkpoint=name,checkpoint_file_sha256=digest.hexdigest(),
    data_dir=DATA_DIR,diagnostic_input=DIAGNOSTIC_INPUT,seed=seed,
    task_classes=classes,client_ids=ids,sample_info=sample_info,partition=partition)
write_json(OUT/'protocol.json',protocol)
'''
    evaluate = '''from tools.eval_denice_matched_oracle import run_matched_oracle
try:
    with threadpool_limits(limits=1):
        summary = run_matched_oracle(ckpt,shards,classes,ids,OUT,device,
                                     DIAGNOSTIC_INPUT,batch_size=BATCH_SIZE)
    print(summary.to_string(index=False))
    display(summary)
finally:
    archive = Path('/kaggle/working/denice_matched_oracle_diagnostics.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file() and 'checkpoints' not in p.relative_to(OUT).parts:
                z.write(p,str(p.relative_to(OUT)))
    print('Diagnostic output (may be partial if execution failed):',archive)
'''
    notebook = dict(nbformat=4,nbformat_minor=5,metadata=prior['metadata'],cells=[
        cell('markdown','# DeNICE Matched Oracle / BestAllowedRoute\n\n'
             'Attach the original 100-clients dataset **and the previous denice_tip_diagnostics.zip** '
             '(or its extracted contents). Enable Internet. The notebook downloads results (4).zip '
             'from the old Drive link. Frozen evaluation only: no training or TIP refit.\n\n'
             'Predictions, task availability and sample identities are replayed from the prior experiment. '
             'Oracle labels are used only in diagnostic selection. Bounds apply only to the enumerated '
             'route actions on this frozen checkpoint and sampled panel.\n'),
        cell('code',setup+locate),cell('code',download),cell('code',prepare),cell('code',evaluate)])
    for index,item in enumerate(notebook['cells']): item['id']=f'matched-{index}'
    path=ROOT/'eval_denice_matched_oracle_kaggle.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(path)


if __name__=='__main__': main()
