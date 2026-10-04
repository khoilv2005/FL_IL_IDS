"""Generate the recorded-graph peer coverage diagnostic notebook."""
import json
from pathlib import Path
from build_denice_tip_notebook import cell

ROOT=Path(__file__).resolve().parents[1]


def main():
    notebook=json.loads((ROOT/'eval_denice_multiclass_integration_kaggle.ipynb').read_text(encoding='utf-8'))
    notebook['cells'][0]=cell('markdown','# DeNICE peer-assisted coverage diagnostic\n\n'
        'Attach the original 100-clients dataset and enable Internet. Checkpoint results (4) '
        'and prior diagnostics are downloaded from the existing Drive links.\n\n'
        'Uses the frozen task-5 round-19 directed graph and positive recorded peer weights. '
        'Separates peer class-mask union on the local model from label-assisted peer expert '
        'selection. Neither training nor graph modification is performed. One GPU is used. '
        'Peer expert enumeration costs substantially more inference than previous notebooks.\n')
    for item in notebook['cells'][1:4]:
        source=''.join(item['source']).replace('denice_multiclass_03b9b53','denice_peer_coverage_03b9b53').replace('denice_multiclass_source','denice_peer_coverage_source')
        source=source.replace('multiclass integration: saved binary memory fit, normal pred_hard inference',
                              'recorded-graph peer mask and peer expert oracle coverage diagnostics')
        item['source']=source.splitlines(keepends=True)
    notebook['cells'][4]=cell('code','''from tools.eval_denice_peer_coverage import run_peer_coverage
try:
    with threadpool_limits(limits=1):
        result=run_peer_coverage(ckpt,shards,classes,ids,OUT,device,
                                 DIAGNOSTIC_INPUT,batch_size=BATCH_SIZE)
    print(result.to_string(index=False))
    display(result)
finally:
    archive=Path('/kaggle/working/denice_peer_coverage_diagnostics.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file() and 'checkpoints' not in p.relative_to(OUT).parts:
                z.write(p,str(p.relative_to(OUT)))
    print('Output (partial if execution failed):',archive)
''')
    for index,item in enumerate(notebook['cells']):item['id']=f'peer-{index}'
    path=ROOT/'eval_denice_peer_coverage_kaggle.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(path)


if __name__=='__main__':main()
