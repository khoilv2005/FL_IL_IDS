"""Build the retrospective, validation-selected competence gate notebook."""
import json
from pathlib import Path
from build_denice_tip_notebook import cell

ROOT=Path(__file__).resolve().parents[1]


def main():
    notebook=json.loads((ROOT/'eval_denice_peer_voting_kaggle.ipynb').read_text(encoding='utf-8'))
    notebook['cells'][0]=cell('markdown','# Frozen DeNICE competence gate diagnostic\n\n'
        'Attach the original 100-clients dataset and enable Internet. Downloads the original '
        'results (4) checkpoint and frozen diagnostic reference using existing Drive links.\n\n'
        'Uses fixed random candidates (seed 42), k=4/8/16 plus self. A shared Logistic Regression '
        'and small MLP estimate expert correctness. Calibration, gate fitting and validation '
        'use content-disjoint original local training rows with recorded task participation. '
        'No test label enters fitting, calibration or policy selection. Gates are saved and locked '
        'before the 50k test experts execute.\n\n'
        '**Retrospective diagnostic:** historical shards are revisited; backbone may have seen '
        'gate validation rows. Shared gate fitting is centralized in this evaluation runtime, '
        'and has not been implemented as a decentralized deployment protocol. All expert '
        'models run where the inputs reside; no raw samples are sent to remote peers. '
        'Expert weights, routes, masks and graph remain frozen. One GPU is used.\n')
    for item in notebook['cells'][1:4]:
        source=''.join(item['source']).replace('denice_peer_voting_03b9b53','denice_competence_gate_03b9b53').replace(
            'denice_peer_voting_source','denice_competence_gate_source').replace(
            'fixed-budget label-free peer inference and separate oracle curves','retrospective validation-selected shared competence gate')
        item['source']=source.splitlines(keepends=True)
    notebook['cells'][4]=cell('code','''from tools.eval_denice_competence_gate import run_competence_gate
from IPython.display import display,Image
PEER_BUDGETS=(4,8,16)
CANDIDATE_SEED=42 # fixed beforehand; not selected using test results
GATE_LIMITS=dict(calibration=128,fit=512,validation=256) # per receiver, distinct content roles
try:
    with threadpool_limits(limits=1):
        result=run_competence_gate(ckpt,shards,classes,ids,OUT,device,DIAGNOSTIC_INPUT,
            DATA_DIR,batch_size=BATCH_SIZE,budgets=PEER_BUDGETS,
            candidate_seed=CANDIDATE_SEED,limits=GATE_LIMITS)
    display(result)
    print(result.to_string(index=False))
    display(Image(filename=str(OUT/'competence_gate_vs_budget.png')))
finally:
    archive=Path('/kaggle/working/denice_competence_gate_diagnostics.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file() and 'checkpoints' not in p.relative_to(OUT).parts:
                z.write(p,str(p.relative_to(OUT)))
    print('Output (partial if execution failed):',archive)
''')
    for index,item in enumerate(notebook['cells']):item['id']=f'competence-gate-{index}'
    path=ROOT/'eval_denice_competence_gate_kaggle.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(path)


if __name__=='__main__':main()
