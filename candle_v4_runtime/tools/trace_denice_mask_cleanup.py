"""Locate protected-row storage changes inside pruning versus Adam steps.

Resumes a trusted small-run continuation, never an original historical archive.
"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fed_learning.clients.denice_client import DeNICEClient
from fed_learning.clients import nice_client
from fed_learning.strategies.decentralized.denice_aggregation import build_compatible_mask
from fed_learning.training import decentralized_denice_il as runner
from tools.diagnose_denice_group import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--age-merge-policy', choices=['consensus','none'],default='consensus')
    args=parser.parse_args()
    torch.set_num_threads(1)
    records=[]
    original_train=DeNICEClient.train
    def train(client,*a,**kw):
        model=client.model
        def capture():
            state=runner._cpu_state_dict(model)
            frozen=build_compatible_mask(state,model.unit_ranks)
            effective={}
            for name,value in state.items():
                layer=name.split('.')[0]
                if layer=='gru':continue  # GRU has output, not connection-weight masks.
                if name.endswith('.weight') and layer in model.weight_masks:
                    effective[name]=value*model.weight_masks[layer].cpu()
                elif name.endswith('.bias') and layer in model.bias_masks:
                    effective[name]=value*model.bias_masks[layer].cpu()
            return state,frozen,effective
        def wrap(original,operation):
            def execute(*aa,**kk):
                before,mask,effective_before=capture()
                value=original(*aa,**kk)
                after,_,effective_after=capture()
                storage=[]; effective=[]
                for name in before:
                    frozen=mask[name]==0
                    if frozen.any():
                        delta=float((after[name].double()-before[name].double())[frozen].abs().max())
                        if delta:storage.append({'name':name,'max_abs_change':delta})
                        if name in effective_before:
                            effective.append(float((effective_after[name]-effective_before[name])[frozen].abs().max()))
                records.append({'client':client.client_id,'task':int(client.y_train[0])//6,
                    'round':kw['phase_offset'],'operation':operation,'changed_protected_storage':storage,
                    'effective_masked_weight_max_change':max(effective,default=0)})
                return value
            return execute
        with patch.object(nice_client,'drop_young_to_learner',wrap(nice_client.drop_young_to_learner,'prune')), \
             patch.object(torch.optim.Adam,'step',wrap(torch.optim.Adam.step,'adam_step')):
            return original_train(client,*a,**kw)
    try:
        with patch.object(DeNICEClient,'train',train):
            run(args.root/'panel',args.output,42,'peer',3,
                resume_state_path=str((args.root/'42_peer/training/continuation_state_task_0.pt').resolve()),task_end=1,
                extra_config={'denice_age_merge_policy':args.age_merge_policy})
    finally:
        if args.output.exists():
            result={'records':records,'summary':{op:{
                'calls':sum(r['operation']==op for r in records),
                'max_protected_storage_change':max((v['max_abs_change'] for r in records if r['operation']==op for v in r['changed_protected_storage']),default=0),
                'max_effective_masked_weight_change':max((r['effective_masked_weight_max_change'] for r in records if r['operation']==op),default=0)} for op in ['prune','adam_step']}}
            (args.output/'mask_trace.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
