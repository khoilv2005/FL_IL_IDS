"""Explicit, provenance-preserving negative-profile transfer during exact bootstrap.

Transfers summaries, never raw examples. Receiver ownership and originating
local participation remain separate. Compatibility is functional, not just equal
feature width. These summaries are not an old-class population FAR certificate.
"""
import copy
import hashlib
import json
import numpy as np
from .config import Rejected
from .input_density_memory import InputDensityMemory
from .mature_profiles import MatureClassProfiles
from .mature_routing import MatureRoutingBranch
from .state import digest

VERSION='appliance_compatible_bootstrap_profile_bundle_v1'
MAX_BYTES=256*1024


def _event(event,sender,receiver,task):
    allowed={'client_id','bootstrap_policy','bootstrap_source','param_distance_to_source','param_distance_to_initial','novelty_bootstrap','novelty_source_has_history'}
    if not isinstance(event,dict) or not set(event).issubset(allowed) or any(isinstance(v,(dict,list,tuple)) for v in event.values()):
        raise Rejected('UNEXPECTED_PROFILE_BOOTSTRAP_PAYLOAD')
    if (type(sender) is not int or type(receiver) is not int or sender==receiver or
            type(task) is not int or not 1<=task<6 or
            event.get('client_id')!=receiver or event.get('bootstrap_source')!=sender or
            event.get('bootstrap_policy')!='representative_clone' or event.get('param_distance_to_source')!=0):
        raise Rejected('INVALID_PROFILE_BOOTSTRAP_AUTHORITY')


def _clean_profile(state):
    obj=MatureClassProfiles.restore(state)
    if obj.state()!=state or any(set(v)!={'task','count','mean','variance'} for v in state['moments']['entries'].values()):
        raise Rejected('UNEXPECTED_PROFILE_EVIDENCE_PAYLOAD')
    branch_keys={'version','rows','input_shape','birth_task','preprocessing_sha256','runtime','dependency_parameter_value_sha','dependency_schema','extra_backbones','extra_parameters_frozen'}
    if set(state['branch'])!=branch_keys:
        raise Rejected('UNEXPECTED_PROFILE_BRANCH_PAYLOAD')
    provenance_keys={'client_id','task','role','role_manifest_sha256','partition_sha256','fit_row_ids_sha256','feature_space_sha256','local_router_refresh_task','local_base_counts','supported_class_ids','summarized_class_ids','fit_rows','event_id'}
    if any(set(p)!=provenance_keys for p in state['provenance']):
        raise Rejected('UNEXPECTED_PROFILE_PROVENANCE_PAYLOAD')
    for p in state['provenance']:
        for key in ('supported_class_ids','summarized_class_ids'):
            values=p[key]
            if not isinstance(values,list) or any(type(c) is not int or not 0<=c<34 for c in values) or values!=sorted(set(values)):
                raise Rejected('INVALID_PROFILE_CLASS_LIST')
    return obj


def _profile(state,model,preprocessing_sha256,task):
    obj=_clean_profile(state)
    if not obj.moments.entries:raise Rejected('NO_GENUINE_LOCAL_CLASS_SUMMARIES')
    if obj.moments.task>=task or obj.moments.task<0:
        raise Rejected('CURRENT_OR_FUTURE_PROFILE_TRANSFER_FORBIDDEN')
    obj.branch.validate(model,preprocessing_sha256)
    return obj


def compile_bundle(sender,receiver,task,event,model,owned=None,imported=None,preprocessing_sha256=None):
    _event(event,sender,receiver,task)
    evidence=[]
    hops=[]
    if owned is not None:
        obj=_profile(owned,model,preprocessing_sha256,task)
        if obj.client_id!=sender:raise Rejected('PROFILE_ORIGIN_NOT_SENDER')
        evidence.append(obj.state())
    if imported is not None:
        cache=ImportedClassProfiles.restore(imported)
        if cache.receiver!=sender:raise Rejected('RELAY_OWNER_CHANGED')
        cache.validate(model,preprocessing_sha256)
        evidence.extend(copy.deepcopy(cache.evidence))
        hops.extend(copy.deepcopy(cache.hops))
    unique={digest(s):s for s in evidence}
    if not unique:raise Rejected('NO_GENUINE_PROFILE_EVIDENCE')
    evidence=[unique[k] for k in sorted(unique)]
    if len({s['role_manifest_sha256'] for s in evidence})!=1:
        raise Rejected('PROFILE_ROLE_LOCK_CHANGED')
    for state in evidence:_profile(state,model,preprocessing_sha256,task)
    hops.append(dict(sender=sender,receiver=receiver,task=task,event=copy.deepcopy(event)))
    payload=dict(version=VERSION,sender=sender,receiver=receiver,bootstrap_task=task,
        evidence_profiles=evidence,hops=hops,retained_raw_examples=0,
        local_participation_claimed=False,old_class_safety_certified=False,main_install_authorized=False)
    packet=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
    if len(packet)>MAX_BYTES:raise Rejected('PROFILE_PACKET_BUDGET_EXCEEDED')
    return packet


class ImportedClassProfiles:
    def __init__(self,receiver,evidence,hops,packet_ids):
        self.receiver=receiver;self.evidence=copy.deepcopy(evidence);self.hops=copy.deepcopy(hops)
        self.packet_ids=list(packet_ids)

    @classmethod
    def accept(cls,packet,model,receiver,task,event,role_manifest,role_sha256,task_classes,existing=None):
        if not isinstance(packet,bytes) or len(packet)>MAX_BYTES:raise Rejected('PROFILE_PACKET_BUDGET_EXCEEDED')
        try:body=json.loads(packet.decode('utf-8'))
        except (UnicodeDecodeError,ValueError) as exc:raise Rejected('INVALID_PROFILE_PACKET') from exc
        allowed_keys={'version','sender','receiver','bootstrap_task','evidence_profiles','hops','retained_raw_examples','local_participation_claimed','old_class_safety_certified','main_install_authorized'}
        if not isinstance(body,dict) or set(body)!=allowed_keys:raise Rejected('UNEXPECTED_PROFILE_PACKET_FIELDS')
        if body.get('version')!=VERSION or body.get('receiver')!=receiver or body.get('bootstrap_task')!=task:
            raise Rejected('PROFILE_PACKET_SESSION_CHANGED')
        _event(event,body['sender'],receiver,task)
        if (body.get('retained_raw_examples')!=0 or body.get('local_participation_claimed') is not False or
                body.get('old_class_safety_certified') is not False or body.get('main_install_authorized') is not False):
            raise Rejected('UNSUPPORTED_IMPORTED_PROFILE_CLAIM')
        hops=body['hops']
        if not hops or hops[-1]!=dict(sender=body['sender'],receiver=receiver,task=task,event=event):
            raise Rejected('PROFILE_RELAY_LAST_HOP_CHANGED')
        previous=set()
        for hop in hops:
            if not isinstance(hop,dict) or set(hop)!={'sender','receiver','task','event'}:
                raise Rejected('UNEXPECTED_PROFILE_RELAY_FIELDS')
            _event(hop['event'],hop['sender'],hop['receiver'],hop['task'])
            if hop['task']>task:raise Rejected('FUTURE_PROFILE_RELAY')
            key=(hop['sender'],hop['receiver'],hop['task'])
            if key in previous:raise Rejected('PROFILE_RELAY_REPLAYED')
            previous.add(key)
        evidence=body['evidence_profiles']
        if not evidence:raise Rejected('NO_GENUINE_PROFILE_EVIDENCE')
        spaces=set()
        for state in evidence:
            obj=MatureClassProfiles.restore(state)
            if obj.role_sha256!=role_sha256:raise Rejected('PROFILE_ROLE_LOCK_CHANGED')
            if obj.branch.state()['preprocessing_sha256']!=role_manifest['metadata_sha256']:
                raise Rejected('IMPORTED_PROFILE_PREPROCESSING_CHANGED')
            origin=str(obj.client_id)
            if origin not in role_manifest['clients']:raise Rejected('UNKNOWN_PROFILE_ORIGIN')
            base=role_manifest['clients'][origin]['role_class_counts']['base']
            for p in obj.provenance:
                classes=list(map(int,task_classes[str(p['task'])]))
                expected={str(c):int(base.get(str(c),0)) for c in classes}
                if p['local_base_counts']!=expected or not set(p['summarized_class_ids']).issubset(classes):
                    raise Rejected('IMPORTED_PROFILE_LOCAL_PROVENANCE_CHANGED')
            _profile(state,model,obj.branch.state()['preprocessing_sha256'],task)
            # A donor origin must reach this receiver through the recorded hops.
            reachable={obj.client_id}
            for hop in hops:
                if hop['sender'] in reachable:reachable.add(hop['receiver'])
            if receiver not in reachable:raise Rejected('PROFILE_RELAY_ORIGIN_UNREACHABLE')
            spaces.add(digest({k:obj.branch.state()[k] for k in (
                'rows','input_shape','preprocessing_sha256','runtime','dependency_parameter_value_sha','dependency_schema')}))
        if len(spaces)!=1:raise Rejected('MIXED_PROFILE_FEATURE_FUNCTIONS')
        packet_id=hashlib.sha256(packet).hexdigest()
        if existing is not None:
            prior=cls.restore(existing)
            if prior.receiver!=receiver or packet_id in prior.packet_ids:raise Rejected('IMPORTED_PROFILE_REPLAYED')
            raise Rejected('IMPORTED_BOOTSTRAP_PROFILE_ALREADY_INITIALIZED')
        packet_ids=[packet_id]
        unique={digest(s):s for s in evidence}
        result=cls(receiver,[unique[k] for k in sorted(unique)],hops,packet_ids)
        result=cls.restore(result.state())
        result.validate(model,evidence[0]['branch']['preprocessing_sha256'])
        return result

    def state(self):
        return dict(version=VERSION,receiver=self.receiver,evidence_profiles=copy.deepcopy(self.evidence),
            hops=copy.deepcopy(self.hops),packet_ids=list(self.packet_ids),retained_raw_examples=0,
            local_participation_claimed=False,old_class_safety_certified=False,main_install_authorized=False)

    @classmethod
    def restore(cls,state):
        allowed_state={'version','receiver','evidence_profiles','hops','packet_ids','retained_raw_examples','local_participation_claimed','old_class_safety_certified','main_install_authorized'}
        if not isinstance(state,dict) or set(state)!=allowed_state:raise Rejected('UNEXPECTED_IMPORTED_PROFILE_STATE_FIELDS')
        if (state.get('version')!=VERSION or type(state.get('receiver')) is not int or state['receiver']<0 or not state.get('evidence_profiles') or
                state.get('retained_raw_examples')!=0 or state.get('local_participation_claimed') is not False or
                state.get('old_class_safety_certified') is not False or state.get('main_install_authorized') is not False or
                not state.get('packet_ids') or len(set(state['packet_ids']))!=len(state['packet_ids'])):
            raise Rejected('INVALID_IMPORTED_PROFILE_STATE')
        evidence=[_clean_profile(s) for s in state['evidence_profiles']]
        if len({digest(s) for s in state['evidence_profiles']})!=len(evidence):
            raise Rejected('DUPLICATE_IMPORTED_PROFILE_EVIDENCE')
        if any(not isinstance(p,str) or len(p)!=64 or any(c not in '0123456789abcdef' for c in p) for p in state['packet_ids']):
            raise Rejected('INVALID_IMPORTED_PROFILE_PACKET_ID')
        for hop in state['hops']:
            if not isinstance(hop,dict) or set(hop)!={'sender','receiver','task','event'}:
                raise Rejected('UNEXPECTED_PROFILE_RELAY_FIELDS')
            _event(hop['event'],hop['sender'],hop['receiver'],hop['task'])
        if not state['hops'] or state['hops'][-1]['receiver']!=state['receiver']:
            raise Rejected('IMPORTED_PROFILE_RESTORE_OWNER_CHANGED')
        tasks=[h['task'] for h in state['hops']]
        if tasks!=sorted(tasks):raise Rejected('NONCHRONOLOGICAL_IMPORTED_PROFILE_HOPS')
        for obj in evidence:
            if obj.moments.task>=tasks[-1]:raise Rejected('CURRENT_OR_FUTURE_PROFILE_TRANSFER_FORBIDDEN')
            reachable={obj.client_id}
            for hop in state['hops']:
                if hop['sender'] in reachable:reachable.add(hop['receiver'])
            if state['receiver'] not in reachable:raise Rejected('PROFILE_RELAY_ORIGIN_UNREACHABLE')
        return cls(state['receiver'],state['evidence_profiles'],state['hops'],state['packet_ids'])

    def validate(self,model,preprocessing_sha256):
        spaces=set()
        for state in self.evidence:
            obj=_clean_profile(state)
            obj.branch.validate(model,preprocessing_sha256)
            spaces.add(digest({k:obj.branch.state()[k] for k in (
                'rows','input_shape','preprocessing_sha256','runtime','dependency_parameter_value_sha','dependency_schema')}))
        if len(spaces)!=1:raise Rejected('MIXED_PROFILE_FEATURE_FUNCTIONS')
        return True

    def negative_memory(self,owned,model,preprocessing_sha256,task):
        if owned.client_id!=self.receiver:raise Rejected('IMPORTED_PROFILE_RECEIVER_CHANGED')
        if type(task) is not int or not 0<=task<6 or owned.moments.task>task or any(h['task']>task for h in self.hops):
            raise Rejected('FUTURE_PROFILE_MEMORY_FORBIDDEN')
        self.validate(model,preprocessing_sha256)
        owned.branch.validate(model,preprocessing_sha256)
        fields=('rows','input_shape','preprocessing_sha256','runtime','dependency_parameter_value_sha','dependency_schema')
        own_key={k:owned.branch.state()[k] for k in fields}
        if any({k:s['branch'][k] for k in fields}!=own_key for s in self.evidence):
            raise Rejected('OWNED_IMPORTED_PROFILE_SPACE_CHANGED')
        result=InputDensityMemory.restore(owned.moments.state())
        offers={}
        for state in self.evidence:
            origin=MatureClassProfiles.restore(state)
            for c,v in origin.moments.entries.items():
                if model.unit_ranks['fc2'][c]<2:continue
                offers.setdefault(c,[]).append((int(v['count']),origin.client_id,copy.deepcopy(v)))
        provenance={}
        for c,values in offers.items():
            if c in result.entries:continue  # Own FIT summary always wins.
            count,source,v=sorted(values,key=lambda x:(-x[0],x[1]))[0]
            result.entries[c]=v
            provenance[str(c)]=dict(origin_client=source,source_task=v['task'],count=count,
                local_participation_claimed=False)
        result.task=max([result.task]+[v['task'] for v in result.entries.values()])
        return result,provenance
