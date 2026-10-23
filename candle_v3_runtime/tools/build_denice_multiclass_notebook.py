"""Generate the normal-pipeline multiclass reproduction notebook."""
import json
from pathlib import Path
from build_denice_tip_notebook import cell

ROOT=Path(__file__).resolve().parents[1]


def main():
    notebook=json.loads((ROOT/'eval_denice_matched_oracle_kaggle.ipynb').read_text(encoding='utf-8'))
    notebook['cells'][0]=cell('markdown','# DeNICE multiclass integration gate\n\n'
        'Attach the original 100-clients dataset and enable Internet. Both results (4).zip '
        'and prior TIP diagnostics are downloaded from the existing Drive links.\n\n'
        'Fits balanced logistic regression using saved binary context memory, then runs normal '
        'pred_hard inference. No backbone training, historical raw train loading or forced oracle routes. '
        'Checks the 27.77% reproduction gate and serialized ContextDetector restore.\n')
    for item in notebook['cells'][1:4]:
        source=''.join(item['source']).replace('denice_matched_03b9b53','denice_multiclass_03b9b53').replace('denice_matched_source','denice_multiclass_source')
        source=source.replace('matched oracle and best allowed route; no router fitting',
                              'multiclass integration: saved binary memory fit, normal pred_hard inference')
        item['source']=source.splitlines(keepends=True)
    notebook['cells'][4]=cell('code','''from tools.eval_denice_multiclass_integration import run_multiclass_integration
try:
    with threadpool_limits(limits=1):
        result=run_multiclass_integration(ckpt,shards,classes,ids,OUT,device,
                                          DIAGNOSTIC_INPUT,batch_size=BATCH_SIZE)
    print(json.dumps(result,indent=2))
finally:
    archive=Path('/kaggle/working/denice_multiclass_integration_diagnostics.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file() and 'checkpoints' not in p.relative_to(OUT).parts:
                z.write(p,str(p.relative_to(OUT)))
    print('Output (partial if execution failed):',archive)
''')
    for index,item in enumerate(notebook['cells']): item['id']=f'multiclass-{index}'
    path=ROOT/'eval_denice_multiclass_integration_kaggle.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(path)


if __name__=='__main__': main()
