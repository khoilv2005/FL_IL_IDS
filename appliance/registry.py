"""Pin live dependencies or invalidate installed knowledge before continued use."""
import copy
import numpy as np
import torch
from .state import boundary_hash,tensor_hash,digest


class Registry:
    def __init__(self):self.entries={}

    def register(self,patch_id,model,detector,class_id,task,pins):
        from fed_learning.training.checkpoint_state import snapshot_context_detector
        values={n:p.detach().cpu().clone() for n,p in model.named_parameters() if pins[n].any()}
        buffers={n:v.detach().cpu().clone() for n,v in model.named_buffers()}
        self.entries[patch_id]=dict(class_id=class_id,task=task,pins=copy.deepcopy(pins),values=values,buffers=buffers,
            backbone_hash=boundary_hash(model,False),
            weight_masks=copy.deepcopy(model.weight_masks),bias_masks=copy.deepcopy(model.bias_masks),
            gru_connection_masks=copy.deepcopy(getattr(model,'gru_connection_masks',{})),
            adapter_input_masks=copy.deepcopy(getattr(model,'adapter_input_masks',{})),
            adapter_output_masks=copy.deepcopy(getattr(model,'adapter_output_masks',{})),
            ranks=copy.deepcopy(model.unit_ranks),valid=True,route_profile=digest(detector.activation_memory.get(task)),
            route_state_hash=digest(snapshot_context_detector(detector)))

    @torch.no_grad()
    def protect(self,model,optimizer=None):
        params=dict(model.named_parameters());buffers=dict(model.named_buffers())
        for e in self.entries.values():
            if not e['valid']:continue
            for name,value in e['values'].items():
                mask=e['pins'][name].to(params[name].device)
                params[name][mask]=value.to(params[name].device)[mask]
                if optimizer is not None:
                    for slot in optimizer.state.get(params[name],{}).values():
                        if torch.is_tensor(slot) and slot.shape==params[name].shape:slot[mask.to(slot.device)]=0
            for name,value in e['buffers'].items():buffers[name].copy_(value.to(buffers[name].device))
            # Restore prefix/selected linear mask coordinates; leave unrelated head rows alone.
            for layer,value in e['weight_masks'].items():
                if layer not in ('fc1','fc2'):
                    model.weight_masks[layer]=value.clone().to(model.weight_masks[layer].device)
                    model.bias_masks[layer]=e['bias_masks'][layer].clone().to(model.bias_masks[layer].device)
                else:
                    mask=e['pins'][f'{layer}.weight']
                    model.weight_masks[layer][mask.to(model.weight_masks[layer].device)]=value.to(model.weight_masks[layer].device)[mask.to(model.weight_masks[layer].device)]
                    bm=e['pins'][f'{layer}.bias']
                    model.bias_masks[layer][bm.to(model.bias_masks[layer].device)]=e['bias_masks'][layer].to(model.bias_masks[layer].device)[bm.to(model.bias_masks[layer].device)]
            model.gru_connection_masks=copy.deepcopy(e['gru_connection_masks'])
            model.adapter_input_masks=copy.deepcopy(e['adapter_input_masks'])
            model.adapter_output_masks=copy.deepcopy(e['adapter_output_masks'])
            for layer,ranks in e['ranks'].items():
                if layer in ('fc1','fc2'):
                    selected=e['pins'][f'{layer}.bias'].numpy()
                    model.unit_ranks[layer][selected]=ranks[selected]
                else:model.unit_ranks[layer]=ranks.copy()

    def gradient_filter(self,model):
        model.reset_frozen_gradients()
        for e in self.entries.values():
            if e['valid']:
                for name,param in model.named_parameters():
                    if param.grad is not None:param.grad[e['pins'][name].to(param.device)]=0

    def validate(self,model,detector,ledger):
        from fed_learning.training.checkpoint_state import snapshot_context_detector
        result={};params=dict(model.named_parameters())
        for patch_id,e in self.entries.items():
            valid=e['valid'] and boundary_hash(model,False)==e['backbone_hash']
            for name,value in e['values'].items():
                mask=e['pins'][name]
                valid=valid and torch.equal(params[name].detach().cpu()[mask],value[mask])
            for layer in ('fc1','fc2'):
                for field,suffix in (('weight_masks','weight'),('bias_masks','bias')):
                    mask=e['pins'][f'{layer}.{suffix}']
                    value=getattr(model,field)[layer].detach().cpu()
                    valid=valid and torch.equal(value[mask],e[field][layer].detach().cpu()[mask])
                selected=e['pins'][f'{layer}.bias'].numpy()
                valid=valid and np.array_equal(model.unit_ranks[layer][selected],e['ranks'][layer][selected])
            valid=valid and e['class_id'] in detector.episode_classes.get(e['task'],[])
            valid=valid and digest(detector.activation_memory.get(e['task']))==e['route_profile']
            valid=valid and digest(snapshot_context_detector(detector))==e['route_state_hash']
            e['valid']=bool(valid)
            if not valid:ledger.invalidate(e['class_id'],'dependency_or_route_drift')
            result[patch_id]=bool(valid)
        return result
