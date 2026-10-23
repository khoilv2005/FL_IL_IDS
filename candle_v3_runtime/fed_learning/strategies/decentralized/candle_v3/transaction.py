"""Shadow head installation and atomic in-process commit/rollback."""
from copy import deepcopy
import torch
from fed_learning.strategies.fed_incremental.nice import update_freeze_masks
from fed_learning.training.checkpoint_state import snapshot_context_detector, restore_context_detector
from .knowledge import state
from .snapshot import digest, reference_payload


def stage_head(reference, fitted):
    model = deepcopy(reference.model)
    targets, phi, permission = fitted['targets'], fitted['head'], fitted['permissions']
    if phi.shape != (len(targets), model.fc2.in_features+1) or not torch.isfinite(phi).all():
        raise ValueError('Invalid fitted head')
    with torch.no_grad():
        for index,c in enumerate(targets):
            if model.unit_ranks['fc2'][c] != 0:
                raise ValueError('Target row is already allocated/protected')
            model.fc2.weight[c].copy_(phi[index,:-1].to(model.fc2.weight))
            model.fc2.bias[c].copy_(phi[index,-1].to(model.fc2.bias))
            coordinate_mask = permission[index] if permission.ndim == 2 else permission
            model.weight_masks['fc2'][c].copy_(coordinate_mask.to(model.weight_masks['fc2']))
            model.bias_masks['fc2'][c] = fitted.get('bias_masks', [1]*len(targets))[index]
            model.unit_ranks['fc2'][c] = 2
    update_freeze_masks(model)
    return model


def commit(live_model, live_detector, reference, staged_model, staged_detector, *, owner, receipt):
    if receipt.get('status') != 'PASS':
        raise ValueError('Only accepted transactions may commit')
    # Hash under the same episode adapter path used to create the reference.
    active = dict(live_model.active_adapters)
    try:
        live_model.set_active_context(reference.identity.task)
        if digest(reference_payload(live_model, live_detector)) != reference.identity.reference_sha256:
            raise ValueError('Receiver reference changed while teaching')
    finally:
        live_model.active_adapters = active
    targets = receipt['targets']
    baseline = {k:v.detach().clone() for k,v in live_model.state_dict().items()}
    ranks = live_model.get_neuron_ages_state()
    masks = live_model.get_masks_state()
    freezes = deepcopy(live_model.freeze_masks)
    registry = deepcopy(getattr(live_model, 'candle_v3_state', None))
    route_before = snapshot_context_detector(live_detector)
    try:
        for key, value in staged_model.state_dict().items():
            if key not in ('fc2.weight','fc2.bias') and not torch.equal(value.cpu(), baseline[key].cpu()):
                raise ValueError('Staging modified a protected non-head tensor')
        for c in range(live_model.num_classes):
            if c not in targets and (not torch.equal(staged_model.fc2.weight[c].cpu(), baseline['fc2.weight'][c].cpu())
                                     or not torch.equal(staged_model.fc2.bias[c].cpu(), baseline['fc2.bias'][c].cpu())):
                raise ValueError('Staging modified another classifier row')
        with torch.no_grad():
            for c in targets:
                if live_model.unit_ranks['fc2'][c] != 0:
                    raise ValueError('Mature target row cannot be overwritten')
                live_model.fc2.weight[c].copy_(staged_model.fc2.weight[c].to(live_model.fc2.weight))
                live_model.fc2.bias[c].copy_(staged_model.fc2.bias[c].to(live_model.fc2.bias))
                live_model.weight_masks['fc2'][c].copy_(staged_model.weight_masks['fc2'][c].to(live_model.weight_masks['fc2']))
                live_model.bias_masks['fc2'][c].copy_(staged_model.bias_masks['fc2'][c].to(live_model.bias_masks['fc2']))
                live_model.unit_ranks['fc2'][c] = staged_model.unit_ranks['fc2'][c]
        update_freeze_masks(live_model)
        restore_context_detector(live_detector, snapshot_context_detector(staged_detector))
        record = state(live_model, owner)
        for c in targets:
            record['peer_acquired'][str(c)] = dict(task=reference.identity.task, snapshot=reference.identity.reference_sha256,
                donors=receipt['donors'], session_id=receipt['session_id'], scope=receipt['scope'],
                evidence=deepcopy(receipt['totals']), protected=True,
                row_sha256=digest([live_model.fc2.weight[c], live_model.fc2.bias[c]]))
    except BaseException:
        live_model.load_state_dict(baseline)
        live_model.set_neuron_ages_state(ranks)
        live_model.set_masks_state(masks)
        live_model.freeze_masks = freezes
        live_model.candle_v3_state = registry
        restore_context_detector(live_detector, route_before)
        raise
