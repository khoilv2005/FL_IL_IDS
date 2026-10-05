"""V2 changes sampling support; keeps V1 experts, candidate pool and gate models."""
from tools.eval_denice_competence_gate import run_competence_gate
from tools.denice_peer_supported_gate_data import peer_supported_data


def run_competence_gate_v2(*args,**kwargs):
    return run_competence_gate(*args,**kwargs,data_builder=peer_supported_data,
        extra_baselines=True,variant='v2_peer_supported')
