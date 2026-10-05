"""Generate V2 peer-supported gate notebook from the existing frozen runner."""
import json
from pathlib import Path
from build_denice_tip_notebook import cell

ROOT=Path(__file__).resolve().parents[1]


def main():
    notebook=json.loads((ROOT/'eval_denice_competence_gate_kaggle.ipynb').read_text(encoding='utf-8'))
    notebook['cells'][0]=cell('markdown','# Frozen DeNICE competence gate V2: peer-supported data\n\n'
        'Attach the original 100-clients dataset and enable Internet. Existing Drive links '
        'download results (4), checkpoint 03b9b53, and the original diagnostic reference.\n\n'
        'Candidate seed 42 and k=4/8/16 are unchanged. Gate calibration/fit/validation now '
        'sample class-balanced rows from self and the fixed maximum-budget candidate origins '
        'with positive graph alpha. Each origin must have its own local task participation '
        'evidence. Inherited binary memory does not authorize historical data access. '
        'Only training-supported classes enter the sampling union; no test-label frequencies '
        'or observed test coverage ratios are used.\n\n'
        'Keep the same LR/MLP architectures, features and hyperparameters. Compare majority, '
        'global donor prior, global donor-task prior, receiver-donor-task prior and learned '
        'gates on the new validation data. Freeze the validation-selected learned policy '
        'before test inference.\n\n'
        '**Development diagnostic, not final untouched test.** The original 50k panel '
        'has informed V2 design. Gate fitting is shared and retrospective; backbone may '
        'have seen gate validation rows. Peer models run locally in this notebook, with '
        'no raw-sample queries to remote peers. A decentralized/private deployment protocol '
        'is not implemented. No backbone, graph, masks or inference router change.\n')
    for item in notebook['cells'][1:4]:
        source=''.join(item['source']).replace('denice_competence_gate_03b9b53','denice_competence_gate_v2_03b9b53').replace(
            'denice_competence_gate_source','denice_competence_gate_v2_source').replace(
            'retrospective validation-selected shared competence gate','V2 peer-supported shared gate development diagnostic')
        item['source']=source.splitlines(keepends=True)
    source=''.join(notebook['cells'][4]['source']).replace(
        'from tools.eval_denice_competence_gate import run_competence_gate',
        'from tools.eval_denice_competence_gate_v2 import run_competence_gate_v2').replace(
        'result=run_competence_gate(','result=run_competence_gate_v2(').replace(
        'denice_competence_gate_diagnostics.zip','denice_competence_gate_v2_diagnostics.zip')
    notebook['cells'][4]=cell('code',source)
    for index,item in enumerate(notebook['cells']):item['id']=f'competence-gate-v2-{index}'
    path=ROOT/'eval_denice_competence_gate_v2_kaggle.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(path)


if __name__=='__main__':main()
