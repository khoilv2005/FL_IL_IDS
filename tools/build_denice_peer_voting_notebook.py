"""Generate the fixed-budget label-free peer inference benchmark notebook."""
import json
from pathlib import Path
from build_denice_tip_notebook import cell

ROOT=Path(__file__).resolve().parents[1]


def main():
    notebook=json.loads((ROOT/'eval_denice_peer_coverage_kaggle.ipynb').read_text(encoding='utf-8'))
    notebook['cells'][0]=cell('markdown','# DeNICE peer voting vs budget\n\n'
        'Attach the original 100-clients dataset and enable Internet. Existing Drive links '
        'download results (4) and the frozen diagnostic reference. Uses checkpoint 03b9b53 '
        'and the same 98 receivers / 50,000 test samples.\n\n'
        'Peer budgets: 0/1/2/4/8/16, plus self. Top-alpha and random peer selection; '
        'majority, alpha and alpha-times-router-confidence votes. Oracle curves are '
        'label-assisted diagnostics only. All settings are fixed before inference.\n\n'
        'One GPU is used. Expert predictions are cached and reused; enumerating oracle '
        'routes increases total benchmark runtime. No backbone training or graph change.\n')
    for item in notebook['cells'][1:4]:
        text=''.join(item['source']).replace('denice_peer_coverage_03b9b53','denice_peer_voting_03b9b53').replace('denice_peer_coverage_source','denice_peer_voting_source')
        text=text.replace('recorded-graph peer mask and peer expert oracle coverage diagnostics',
                          'fixed-budget label-free peer inference and separate oracle curves')
        item['source']=text.splitlines(keepends=True)
    notebook['cells'][4]=cell('code','''from tools.eval_denice_peer_voting import run_peer_voting
PEER_BUDGETS=(0,1,2,4,8,16)
RANDOM_PEER_SEEDS=(42,43,44,45,46)
try:
    with threadpool_limits(limits=1):
        result=run_peer_voting(ckpt,shards,classes,ids,OUT,device,DIAGNOSTIC_INPUT,
            batch_size=BATCH_SIZE,budgets=PEER_BUDGETS,random_seeds=RANDOM_PEER_SEEDS)
    print(result.to_string(index=False))
    display(result)
    from IPython.display import Image,display
    display(Image(filename=str(OUT/'accuracy_oracle_vs_budget.png')))
finally:
    archive=Path('/kaggle/working/denice_peer_voting_diagnostics.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file() and 'checkpoints' not in p.relative_to(OUT).parts:
                z.write(p,str(p.relative_to(OUT)))
    print('Output (partial if execution failed):',archive)
''')
    for index,item in enumerate(notebook['cells']):item['id']=f'peer-vote-{index}'
    path=ROOT/'eval_denice_peer_voting_kaggle.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(path)


if __name__=='__main__':main()
