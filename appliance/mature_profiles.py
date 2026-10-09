"""Receiver-owned class moments in one fixed mature routing feature space.

Raw calibration and per-example features are discarded after summarization.
A summary is not an old-class FAR certificate. Missing pre-birth/unsupported
classes stay explicit; inherited router memory never fills that gap.
"""
import copy
import hashlib
import numpy as np
from .config import Rejected
from .current_calibration_data import CurrentCalibrationData
from .imported_route import ROUTE_RULES, stratified_roles
from .input_density_memory import InputDensityMemory
from .mature_routing import MatureRoutingBranch
from .state import digest


class MatureClassProfiles:
    VERSION = 'appliance_receiver_mature_class_profiles_v1'

    def __init__(self, client_id, branch, role_sha256):
        if type(client_id) is not int or client_id < 0 or not isinstance(branch, MatureRoutingBranch):
            raise Rejected('INVALID_MATURE_PROFILE_OWNER')
        self.client_id = client_id
        self.branch = MatureRoutingBranch(branch.state())
        self.role_sha256 = role_sha256
        self.space_sha256 = digest(self.branch.state())
        self.moments = InputDensityMemory((len(self.branch.state()['rows']),), self.space_sha256)
        self.provenance = []

    @classmethod
    def capture(cls, model, detector, scoped, width=64):
        cls._participation(model, detector, scoped)
        return cls(scoped.client_id, MatureRoutingBranch.capture(
            model, scoped.store['metadata_sha256'], scoped.task, width=width), scoped.store['role_manifest_sha256'])

    @staticmethod
    def _participation(model, detector, scoped):
        if not isinstance(scoped, CurrentCalibrationData):
            raise Rejected('TASK_SCOPED_CALIBRATION_REQUIRED')
        if getattr(detector, 'router_last_refresh_task', None) != scoped.task:
            raise Rejected('LOCAL_CURRENT_TASK_PARTICIPATION_REQUIRED')
        classes = scoped.store['task_classes'][str(scoped.task)]
        counts = scoped.manifest['clients'][str(scoped.client_id)]['role_class_counts']['base']
        if not any(int(counts.get(str(c), 0)) > 0 for c in classes):
            raise Rejected('LOCAL_BASE_PROVENANCE_REQUIRED')

    def validate(self, model, scoped):
        if (not isinstance(scoped, CurrentCalibrationData) or scoped.client_id != self.client_id or
                scoped.store['role_manifest_sha256'] != self.role_sha256 or
                scoped.task < self.branch.state()['birth_task']):
            raise Rejected('MATURE_PROFILE_PROVENANCE_CHANGED')
        self.branch.validate(model, scoped.store['metadata_sha256'])

    def observe_current(self, model, detector, scoped, batch_size=512):
        self.validate(model, scoped)
        self._participation(model, detector, scoped)
        if any(p['task'] == scoped.task for p in self.provenance):
            raise Rejected('MATURE_PROFILE_TASK_ALREADY_SEALED')
        if scoped.task < self.moments.task:
            raise Rejected('HISTORICAL_MATURE_PROFILE_REOPENED')
        classes = scoped.store['task_classes'][str(scoped.task)]
        pool = scoped.current_pool(self.client_id, 'calibration', classes)
        fit = stratified_roles(pool, ROUTE_RULES['seed'] + self.client_id)['fit']
        counts = scoped.manifest['clients'][str(self.client_id)]['role_class_counts']['base']
        supported = [c for c in classes if int(counts.get(str(c), 0)) > 0 and model.unit_ranks['fc2'][c] >= 2]
        selected = fit[np.isin(pool['y'][fit], supported)]
        features = self.branch.features(model, pool['X'][selected], scoped.store['metadata_sha256'], batch_size)
        row_sha = hashlib.sha256(np.asarray(pool['rows'][selected], dtype='<i8').tobytes()).hexdigest()
        event = digest(dict(owner=self.client_id, task=scoped.task, partition=pool['partition_sha256'],
            rows_sha256=row_sha, space=self.space_sha256, role='current CALIBRATION FIT'))
        pending = InputDensityMemory.restore(self.moments.state())
        pending.observe(scoped.task, classes, features, pool['y'][selected], event)
        provenance = dict(client_id=self.client_id, task=scoped.task, role='current CALIBRATION FIT',
            role_manifest_sha256=self.role_sha256, partition_sha256=pool['partition_sha256'],
            fit_row_ids_sha256=row_sha, feature_space_sha256=self.space_sha256,
            local_router_refresh_task=int(detector.router_last_refresh_task),
            local_base_counts={str(c): int(counts.get(str(c),0)) for c in classes},
            supported_class_ids=supported, summarized_class_ids=sorted(map(int,np.unique(pool['y'][selected]))),
            fit_rows=len(selected), event_id=event)
        self.moments = pending
        self.provenance.append(provenance)
        self.validate(model, scoped)
        return copy.deepcopy(provenance)

    def state(self):
        return dict(version=self.VERSION, client_id=self.client_id, branch=self.branch.state(),
            role_manifest_sha256=self.role_sha256, feature_space_sha256=self.space_sha256,
            moments=self.moments.state(), provenance=copy.deepcopy(self.provenance),
            retained_raw_examples=0, retained_per_sample_features=0,
            old_class_safety_certified=False, main_install_authorized=False)

    @classmethod
    def restore(cls, state):
        if state.get('version') != cls.VERSION:
            raise Rejected('MATURE_PROFILE_VERSION_CHANGED')
        obj = cls(state['client_id'], MatureRoutingBranch(state['branch']), state['role_manifest_sha256'])
        if obj.space_sha256 != state['feature_space_sha256']:
            raise Rejected('MATURE_PROFILE_FEATURE_SPACE_CHANGED')
        obj.moments = InputDensityMemory.restore(state['moments'])
        if obj.moments.input_shape != (len(obj.branch.state()['rows']),) or obj.moments.preprocessing_sha256 != obj.space_sha256 or obj.moments.transform != 'identity':
            raise Rejected('MATURE_PROFILE_MOMENT_SPACE_CHANGED')
        obj.provenance = copy.deepcopy(state['provenance'])
        tasks = [p['task'] for p in obj.provenance]
        if (tasks != sorted(set(tasks)) or any(type(t) is not int or t < obj.branch.state()['birth_task'] or t > 5 for t in tasks) or
                set(obj.moments.events) != {p['event_id'] for p in obj.provenance}):
            raise Rejected('INVALID_MATURE_PROFILE_HISTORY')
        if obj.moments.task != (tasks[-1] if tasks else -1):
            raise Rejected('MATURE_PROFILE_TASK_STATE_CHANGED')
        for p in obj.provenance:
            expected_event=digest(dict(owner=obj.client_id,task=p['task'],partition=p['partition_sha256'],
                rows_sha256=p['fit_row_ids_sha256'],space=obj.space_sha256,role='current CALIBRATION FIT'))
            if p['event_id']!=expected_event:
                raise Rejected('MATURE_PROFILE_EVENT_BINDING_CHANGED')
            if (p['client_id'] != obj.client_id or p['role'] != 'current CALIBRATION FIT' or
                    p['role_manifest_sha256'] != obj.role_sha256 or p['feature_space_sha256'] != obj.space_sha256 or
                    p['local_router_refresh_task'] != p['task']):
                raise Rejected('INVALID_MATURE_PROFILE_PARTICIPATION')
            entries = sorted(c for c,v in obj.moments.entries.items() if v['task'] == p['task'])
            if (entries != p['summarized_class_ids'] or
                    sum(obj.moments.entries[c]['count'] for c in entries) != p['fit_rows'] or
                    any(c not in p['supported_class_ids'] or p['local_base_counts'].get(str(c),0) <= 0 for c in entries)):
                raise Rejected('INVALID_MATURE_PROFILE_CLASS_PROVENANCE')
        if obj.moments.entries and any(v['task'] not in tasks for v in obj.moments.entries.values()):
            raise Rejected('MATURE_PROFILE_WITHOUT_TASK_PROVENANCE')
        if (state.get('retained_raw_examples') != 0 or state.get('retained_per_sample_features') != 0 or
                state.get('old_class_safety_certified') is not False or state.get('main_install_authorized') is not False):
            raise Rejected('UNSUPPORTED_MATURE_PROFILE_CERTIFICATE')
        return obj

    def coverage(self, model, task_classes, through_task):
        required = sorted(c for t, classes in task_classes.items() if int(t) <= through_task
            for c in classes if model.unit_ranks['fc2'][c] >= 2)
        missing = sorted(set(required) - set(self.moments.entries))
        counts={str(c):self.moments.entries[c]['count'] if c in self.moments.entries else 0 for c in required}
        return dict(required_mature_classes=required, summarized_classes=sorted(self.moments.entries),
            missing_classes=missing, complete=not missing, completeness_scope='presence only; not statistical safety',
            summary_counts=counts,classes_below_32_fit_rows=[c for c in required if counts[str(c)]<32],
            old_class_safety_certified=False)
