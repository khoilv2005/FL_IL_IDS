"""Bounded native next-task rounds with FIT-discovered, accepted head patches.

The federation continuation, CANC, aggregation and router refresh are preserved.
Each tracked receiver has one accepted capability. Historical calibration is an
offline diagnostic and is not a replay-free deployment protocol.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile

import torch

from .config import Rejected
from .guarded_head import GuardedHeadRegistry
from .native_lifecycle_experiment import load_continuation,_splice
from .selector import lookup
from .state import write_json


def run_active_native(checkpoint,data_dir,roles_dir,source,out,device='cpu',round_budget=1,batch_size=512,
                      pair_policy='all_accepted'):
    from eval_checkpoint import _make_denice_client_model
    from fed_learning.data.denice_clean_roles import CleanRoleData,file_sha256
    from fed_learning.training.decentralized_denice_il import run_decentralized_denice_il
    source,out=Path(source),Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new output directory')
    out.mkdir(parents=True,exist_ok=True);seed_path=None
    write_json(out/'completion.json',dict(completed_execution=False,passed_feasibility=False,stage='prepare'))
    try:
        protocol=json.loads((source/'protocol_lock.json').read_text(encoding='utf-8'))
        completed=json.loads((source/'completion.json').read_text(encoding='utf-8'))
        if not completed['completed_execution'] or completed['validation_opened'] or completed['final_test_opened']:
            raise Rejected('CALIBRATION_SOURCE_SCOPE_INVALID')
        accepted=[v for v in completed['outcomes'] if v['installed']]
        if not accepted:raise Rejected('ACCEPTED_ACTIVE_PAIRS_REQUIRED')
        if pair_policy=='first_per_class':
            selected={}
            for value in accepted:selected.setdefault(value['class_id'],value)
            accepted=list(selected.values())
        elif pair_policy!='all_accepted':raise ValueError('Unknown pair policy')
        if len({p['receiver'] for p in accepted})!=len(accepted):raise Rejected('ONE_CAPABILITY_PER_RECEIVER_REQUIRED')
        task=int(protocol['task'])
        if any(int(p['task'])!=task or int(p['next_task'])!=task+1 for p in accepted):raise Rejected('PAIR_TASK_SCOPE_CHANGED')
        audit=json.loads((source/'independent_audit.json').read_text(encoding='utf-8'))
        if audit['mismatches'] or audit['accepted_count']!=completed['accepted_count']:raise Rejected('SOURCE_INDEPENDENT_AUDIT_REQUIRED')
        roles=CleanRoleData(roles_dir,source_data_dir=data_dir)
        if file_sha256(roles.root/'role_manifest.json')!=protocol['role_manifest_sha256']:raise Rejected('ROLE_LOCK_CHANGED')
        for i in roles.manifest['clients']:
            if not (roles.root/'indices'/f'client_{i}.npz').is_file():raise Rejected('FULL_FEDERATION_ROLES_REQUIRED',i)
        if any(p['next_task_base_rows']<=0 for p in accepted):raise Rejected('ACTIVE_RECEIVER_REQUIRED')
        state,continuation_hash=load_continuation(checkpoint,protocol['terminal_sha256'],task)
        config=copy.deepcopy(state['config'])
        original_amp=config['denice_amp_enabled']
        if config.get('denice_cl_method')!='legacy' or config['denice_similarity_threshold']!=.8:raise Rejected('LEGACY_XI8_REQUIRED')
        metadata=json.loads((roles.source/'metadata.json').read_text(encoding='utf-8'))
        tasks={int(t):list(map(int,v)) for t,v in metadata['task_structure']['task_classes'].items()}
        portable_dirs={};packet_hashes={}
        # The entire subset is fixed from the pre-existing schedule, before any
        # native measurements. No substitution of failed survival pairs.
        write_json(out/'survival_pair_lock.json',dict(policy=pair_policy,pairs=accepted))
        for pair in accepted:
            cid,c,donor=pair['receiver'],pair['class_id'],pair['donor']
            folder=source/f'receiver_{cid}_class_{c}'
            packet=(folder/'candidate.bin').read_bytes();decision=json.loads((folder/'decision_lock.json').read_text(encoding='utf-8'))
            if hashlib.sha256(packet).hexdigest()!=decision['candidate_packet_sha256']:raise Rejected('ACCEPTED_PACKET_CHANGED')
            packet_hashes[str(cid)]=decision['candidate_packet_sha256']
            seed=torch.load(folder/'guarded_receiver.pt',map_location='cpu',weights_only=False)
            baseline=lookup(state['client_model_states'],cid);patched=lookup(seed['client_model_states'],cid)
            if set(baseline)!=set(patched):raise Rejected('SEED_MODEL_KEYS_CHANGED')
            for name,value in baseline.items():
                changed=value!=patched[name];allowed=torch.zeros_like(changed)
                if name in ('fc2.weight','fc2.bias'):allowed[c]=True
                if bool((changed&~allowed).any()):raise Rejected('SEED_NONHEAD_CHANGE',name)
            model,router=_make_denice_client_model(seed,cid,device)
            registry=GuardedHeadRegistry();registry.entries=seed['guarded_head_entries'];registry.sync(model)
            if not registry.certificate_current(model,router,c):raise Rejected('ACCEPTED_SEED_CERTIFICATE_STALE')
            _splice(state,model,router,cid);del model,router,registry,seed
            portable=out/'frozen_capabilities'/f'receiver_{cid}';portable.mkdir(parents=True)
            calibration=json.loads((folder/'calibration_manifest.json').read_text(encoding='utf-8'))
            write_json(portable/'calibration_split_manifest.json',calibration)
            portable_protocol=dict(protocol,receiver=cid,donor=donor,class_id=c,
                acceptance_class_scope_by_client={str(cid):sorted(k for t in range(task+1) for k in tasks[t]),str(donor):tasks[task]})
            write_json(portable/'protocol_lock.json',portable_protocol);write_json(portable/'pair_lock.json',pair)
            (portable/'candidate.bin').write_bytes(packet)
            portable_dirs[str(cid)]=str(portable.resolve())
        spec=dict(version='appliance_active_native_historical_probe_v1',output_dir=str(out.resolve()),
            roles_dir=str(roles.root.resolve()),portable_dirs=portable_dirs,
            round_budget=int(round_budget),audit_batch_size=int(batch_size),audit_optimizer_delta=True,
            receiver_per_class_far_budget=.001,measure_validation=False)
        spec_path=out/'native_probe_manifest.json';write_json(spec_path,spec)
        config.update(data_dir=str(Path(data_dir).resolve()),denice_clean_roles_dir=str(roles.root.resolve()),
            denice_evaluation_data_role='validation',denice_post_task_eval=False,
            denice_eval_final_round=False,denice_eval_last_round_only=False,denice_eval_terminal_state_only=True,
            denice_eval_local_validation=False,eval_every=999999,denice_cme_after_each_task=False,
            denice_save_round_artifacts=False,round_checkpoint_every=None,save_continuation_every_task=False,
            denice_archive_checkpoints=False,output_dir=str((out/'native_training').resolve()),
            resume_output_dir=str((out/'native_training').resolve()),
            denice_amp_initial_scale=1024.)
        state['config']=config
        handle=tempfile.NamedTemporaryFile(prefix='appliance-active-native-',suffix='.pt',delete=False)
        seed_path=Path(handle.name);handle.close();torch.save(state,seed_path);del state
        config.update(resume_state_path=str(seed_path),appliance_native_probe_manifest=str(spec_path.resolve()))
        write_json(out/'protocol_lock.json',dict(pairs=accepted,pair_policy=pair_policy,source_task=task,
            source_terminal_sha256=protocol['terminal_sha256'],
            source_continuation_sha256=continuation_hash,source_packet_sha256_by_receiver=packet_hashes,
            role_manifest_sha256=protocol['role_manifest_sha256'],method='legacy',xi=.8,
            round_budget=round_budget,original_round_schedule=config['rounds_per_task'],
            training_batch_size=config['batch_size'],numeric_policy='FP32 CPU' if device=='cpu' else 'AMP initial scale 1024',
            original_training_amp=original_amp,training_amp=config['denice_amp_enabled'],
            effective_amp_enabled=bool(config['denice_amp_enabled'] and device.startswith('cuda')),
            entire_federation_preserved=True,optimizer_parameter_delta_audit=True,
            validation_opened=False,final_test_opened=False,thresholds_retuned=False,
            prototype_refitted=False,measurement_role='locked historical calibration HOLDOUT',
            limitation='Bounded native survival of development capabilities; not full-task/full-campaign or replay-free proof'))
        print(f'Native survival: task={task}->{task+1}, pairs={len(accepted)}, rounds={round_budget}, device={device}',flush=True)
        return run_decentralized_denice_il(config)
    except Exception as exc:
        write_json(out/'completion.json',dict(completed_execution=False,passed_feasibility=False,
            error_type=type(exc).__name__,error=str(exc),validation_opened=False,final_test_opened=False))
        raise
    finally:
        if seed_path is not None:seed_path.unlink(missing_ok=True)
