"""Runner-owned automatic APPLIANCE service. No prepared packets or old CAL.

Initial CAL evidence remains scoped. Empirical deployment, when explicitly
selected, permits unchanged accepted guards on all application inputs without
claiming that unobserved classes have been certified. Failed requests and
observed conflicts suspend/skip; they never abort the DeNICE schedule.
"""
import copy
import json
import time
from pathlib import Path

import torch

from .active_pair_calibration import split_counts
from .base_sketch_shield import CurrentBaseSketchShield
from .config import Rejected
from .current_base_data import CurrentBaseData
from .current_calibration_data import CurrentCalibrationData
from .current_installation import install_current_pair
from .current_scope_evidence import plan_current_witnesses, collect_current_receipts
from .cumulative_certificate import VERSION as CUMULATIVE_VERSION, validated_scope
from .empirical_deployment import VERSION as EMPIRICAL_VERSION
from .discovery_preflight import receiver_preflight
from .live_discovery import LiveCurrentClient, discover_live_pairs
from .patch_lifecycle import route_authorized
from .transport import Transport
from .config import Protocol
from .portable_route import SharedSketch
from .stable_head import StableHeadRegistry, POLICY
from .state import digest, rng_snapshot, restore_rng, write_json


class ApplianceTrainingService:
    VERSION = 'appliance_automatic_runner_v1'

    def __init__(self, config, saved=None):
        self.config = config
        self.scope_mode=config.get('appliance_scope_mode','initial_scope_v1')
        if self.scope_mode not in ('initial_scope_v1',CUMULATIVE_VERSION,EMPIRICAL_VERSION):
            raise ValueError('Unknown APPLIANCE scope mode')
        self.cumulative=self.scope_mode==CUMULATIVE_VERSION
        self.empirical=self.scope_mode==EMPIRICAL_VERSION
        self.improved_discovery=self.cumulative or self.empirical
        if (config.get('denice_cl_method') != 'legacy' or
                config.get('denice_cme_after_each_task') or
                config.get('appliance_current_native_manifest') or
                config.get('appliance_native_probe_manifest')):
            raise ValueError('Automatic APPLIANCE requires legacy DeNICE without CME or diagnostic observers')
        self.out = Path(config['output_dir']) / 'appliance'
        self.out.mkdir(parents=True, exist_ok=True)
        self.views, self.base_views, self.shields = {}, {}, {}
        self.rounds, self.communication = [], []
        self.attempts = set()
        self.contract = dict(version=self.VERSION, guard=POLICY,
            role_sha=config['denice_data_roles_sha256'],
            input_shape=list(config['input_shape']),
            batch_size=int(config.get('appliance_batch_size', 512)),
            application_domain=config.get('appliance_application_domain', 'cumulative_dataset'),
            selection='existing current_local_fit selector; one receiver per class',
            multi_capability='skip until independently verified')
        if self.improved_discovery:
            self.contract.update(version='appliance_automatic_runner_v2',scope_mode=self.scope_mode,
                selection='metadata preflight; multiple receivers/class; FIT only; no HOLDOUT substitution',
                max_transactions=int(config.get('appliance_discovery_max_transactions',16)),
                max_receivers_per_class=int(config.get('appliance_discovery_max_receivers_per_class',8)),
                scope_peer_budget=int(config.get('appliance_scope_max_peers',4)))
            if self.empirical:
                self.contract.update(deployment='all application inputs; explicit empirical risk outside CAL evidence',
                    population_FAR_claim=False, observed_conflict_suspends=True)
        if saved:
            if saved['contract'] != self.contract:
                raise ValueError('APPLIANCE resume contract changed')
            self.views = {int(i): CurrentCalibrationData.restore(config['appliance_calibration_store'], s)
                          for i, s in saved['calibration_views'].items()}
            self.base_views = {int(i): CurrentBaseData.restore(config['appliance_base_store'], s)
                               for i, s in saved['base_views'].items()}
            self.shields = {int(i): CurrentBaseSketchShield.restore(s) for i, s in saved['shields'].items()}
            self.rounds = copy.deepcopy(saved['rounds'])
            self.communication = copy.deepcopy(saved['communication'])
            self.attempts = set(saved['attempts'])
        write_json(self.out / 'service_contract.json', self.contract)

    def registry(self, model):
        reg = StableHeadRegistry()
        reg.entries = getattr(model, 'appliance_guarded_head_entries', {})
        return reg

    def protect(self, model):
        reg = self.registry(model)
        if reg.entries:
            reg.protect(model)

    def training_hooks(self, cid, model):
        reg = self.registry(model)
        if not reg.entries:
            return {}
        reg.protect(model)
        optimizers = []

        def factory(parameters, lr):
            optimizer = torch.optim.Adam(parameters, lr=lr)
            reg.protect(model, optimizer)
            optimizers.append(optimizer)
            return optimizer

        return dict(optimizer_factory=factory,
                    gradient_filter=lambda: reg.gradient_filter(model),
                    after_optimizer_step_update=lambda current: reg.protect(current, optimizers[-1]))

    def begin_task(self, task, ids, models):
        rng = rng_snapshot()
        try:
            for cid in ids:
                if cid not in self.views:
                    self.views[cid] = CurrentCalibrationData(self.config['appliance_calibration_store'], cid, task,
                                                            self.contract['role_sha'])
                    self.base_views[cid] = CurrentBaseData(self.config['appliance_base_store'], cid, task,
                                                          self.contract['role_sha'])
                else:
                    # Inactive clients may skip tasks. Advancing metadata opens no raw rows.
                    while self.views[cid].task < task:
                        self.views[cid].advance(self.views[cid].task + 1)
                        self.base_views[cid].advance(self.base_views[cid].task + 1)
                    if self.views[cid].task != task:
                        raise Rejected('RUNNER_CAL_TASK_REGRESSION')
                if cid not in self.shields:
                    saved = getattr(models[cid], 'appliance_base_sketch_shield_state', None)
                    self.shields[cid] = (CurrentBaseSketchShield.restore(saved) if saved else
                        CurrentBaseSketchShield(self.base_views[cid], SharedSketch(tuple(self.config['input_shape']), 16,
                            self.base_views[cid].store['metadata_sha256'])))
                shield = self.shields[cid]
                if shield.provenance and shield.provenance[-1]['task'] > task:
                    raise Rejected('RUNNER_BASE_SUMMARY_TASK_REGRESSION')
                if not shield.provenance or shield.provenance[-1]['task'] < task:
                    shield.observe_current(self.base_views[cid])
                models[cid].appliance_base_sketch_shield_state = shield.state()
                self.protect(models[cid])
        finally:
            restore_rng(rng)

    @staticmethod
    def graph(cluster, ids):
        ids = set(ids)
        groups, alphas = {}, {}
        for cid in sorted(ids):
            item = cluster['alpha_debug'].get(cid, cluster['alpha_debug'].get(str(cid)))
            raw = cluster['groups'].get(cid, cluster['groups'].get(str(cid)))
            donors = list(map(int, item['group_ids']))
            weights = list(map(float, item['alphas']))
            if len(donors) != len(weights) or len(set(donors)) != len(donors) or set(donors) != set(map(int, raw)):
                raise Rejected('RUNNER_GRAPH_MALFORMED')
            alphas[cid] = dict(zip(donors, weights))
            groups[cid] = sorted(d for d, w in alphas[cid].items() if d != cid and w > 0)
        return groups, alphas

    def runtime_scope(self, seen):
        # Declared once for the application pool, never from per-sample labels/tasks.
        return dict(domain_id=self.contract['application_domain'], classes=list(map(int, seen)))

    def _wire_metadata(self, discovery, groups, task, round_id, phase):
        for receiver, donors in groups.items():
            req = [r for r in discovery['requests'] if int(r['receiver']) == receiver]
            for donor in donors:
                offers = [o for o in discovery['offers'] if int(o['donor']) == donor]
                for sender, recipient, kind, body in ((receiver, donor, 'discovery_requests', req),
                                                      (donor, receiver, 'discovery_offers', offers)):
                    data = json.dumps(body, sort_keys=True, separators=(',', ':')).encode()
                    self.communication.append(dict(task=task, round=round_id, phase=phase,
                        sender=sender, receiver=recipient, kind=kind, bytes=len(data)))

    def after_refresh(self, task, round_id, models, routers, clients, ids, cluster, seen, phase='round'):
        key = (int(task), int(round_id), phase)
        if any((r['task'], r['round'], r['phase']) == key for r in self.rounds):
            raise Rejected('RUNNER_CALLBACK_REPLAYED')
        started = time.perf_counter()
        rng = rng_snapshot()
        try:
            groups, alphas = self.graph(cluster, ids)
            endpoints = {cid: LiveCurrentClient(cid, models[cid], routers[cid], self.views[cid], seen,
                                                self.contract['batch_size']) for cid in ids}
            preflight=({cid:receiver_preflight(cid,models[cid],self.views[cid],self.shields[cid],seen)
                        for cid in ids} if self.improved_discovery else None)
            discovery = discover_live_pairs(endpoints, groups, alphas, task, round_id,preflight,
                self.contract.get('max_transactions',16),self.contract.get('max_receivers_per_class',8))
            self._wire_metadata(discovery, groups, task, round_id, phase)
            transactions = []
            # Snapshot all selected donors before the first commit. An installed
            # receiver cannot accidentally become a different later donor offer.
            snapshots = {p['donor']: (copy.deepcopy(models[p['donor']]), copy.deepcopy(routers[p['donor']]))
                         for p in discovery['pairs']}
            for pair in discovery['pairs']:
                cid, donor, c = (int(pair[k]) for k in ('receiver', 'donor', 'class_id'))
                attempt = digest(dict(pair=pair, phase=phase))
                if attempt in self.attempts:
                    transactions.append(dict(pair=pair, transaction=dict(applied=False, reason='identical_request_already_attempted')))
                    continue
                counts = self.views[cid].store['clients'][str(cid)][str(task)]['class_counts']
                scope = dict(domain_id=self.contract['application_domain'],
                             classes=sorted({c} | {int(k) for k, n in counts.items() if split_counts(int(n))[2] > 0}))
                target = self.out / f'task_{task}_round_{round_id}_{phase}' / f'receiver_{cid}_class_{c}'
                candidate, detector, reg, report = install_current_pair(pair, models[cid], routers[cid],
                    *snapshots[donor], self.config, self.views[cid], self.views[donor], self.shields[cid],
                    groups, alphas, seen, scope, target, self.contract['batch_size'], session_nonce=attempt)
                if report['transaction']['applied']:
                    models[cid], routers[cid] = candidate, detector
                    clients[cid].model = candidate
                    reg.sync(candidate)
                    reg.protect(candidate)
                self.attempts.add(attempt)
                wire = target / 'application_wire.jsonl'
                if wire.exists():
                    for line in wire.read_text(encoding='utf-8').splitlines():
                        self.communication.append(dict(task=task, round=round_id, phase=phase, **json.loads(line)))
                transactions.append(report)
                print(f'APPLIANCE install task={task} round={round_id} {donor}->{cid}/class{c}: '
                      f'{report["transaction"]}', flush=True)
            deployments=[]
            if self.empirical:
                for cid in sorted(ids):
                    reg=self.registry(models[cid])
                    for c in sorted(reg.entries):
                        try:
                            declared=reg.bind_empirical_deployment(models[cid],routers[cid],c)
                            deployments.append(dict(receiver=cid,class_id=c,deployment_scope=declared,
                                CAL_scope=validated_scope(reg.entries[c]),population_FAR_claim=False))
                        except Rejected as exc:
                            deployments.append(dict(receiver=cid,class_id=c,reason=exc.reason))
            scope_updates=(self.extend_scopes(task,round_id,phase,models,routers,ids,groups,alphas,seen)
                           if self.cumulative else [])
            lifecycle = self.check_lifecycle(task, models, routers, seen)
            record = dict(task=task, round=round_id, phase=phase, discovery=discovery,
                          transactions=transactions, lifecycle=lifecycle,scope_updates=scope_updates,
                          empirical_deployments=deployments,
                          seconds=time.perf_counter() - started, historical_raw_CAL_reads=0)
            self.rounds.append(record)
            write_json(self.out / 'automatic_history.json', self.rounds)
            write_json(self.out / 'communication.json', self.communication_summary())
            print(f'APPLIANCE automatic task={task} round={round_id} phase={phase}: '
                  f'requests={len(discovery["requests"])} offers={len(discovery["offers"])} '
                  f'commits={sum(r["transaction"]["applied"] for r in transactions)} '
                  f'authorized={sum(route_authorized(e,self.runtime_scope(seen)) for m in models.values() for e in self.registry(m).entries.values())} '
                  f'bytes={self.communication_summary()["application_egress_bytes"]}', flush=True)
            return record
        finally:
            restore_rng(rng)

    def extend_scopes(self,task,round_id,phase,models,routers,ids,groups,alphas,seen):
        reports=[]
        for cid in sorted(ids):
            reg=self.registry(models[cid])
            for c,e in sorted(reg.entries.items()):
                try:
                    scope=validated_scope(e)
                    missing=sorted(set(seen)-set(scope['classes']))
                    if not missing or e.get('new_conflict_latched'):
                        reports.append(dict(receiver=cid,class_id=c,issued=False,
                            reason='conflict_latched' if e.get('new_conflict_latched') else 'scope_already_covered'))
                        continue
                    if not reg.certificate_current(models[cid],routers[cid],c):
                        raise Rejected('SCOPE_EXTENSION_FUNCTION_CHANGED')
                    classes=self.views[cid].store['task_classes'][str(task)]
                    plan=plan_current_witnesses(cid,e,task,classes,groups,alphas,self.views,
                        self.contract['scope_peer_budget'])
                    if not plan['missing_current_classes']:
                        reports.append(dict(receiver=cid,class_id=c,issued=False,
                            reason='missing_prior_classes_no_historical_CAL_authority',missing_classes=missing))
                        continue
                    if not plan['owners']:
                        reports.append(dict(receiver=cid,class_id=c,issued=False,
                            reason='no_current_witnesses',plan=plan))
                        continue
                    directory=self.out/f'task_{task}_round_{round_id}_{phase}'/f'scope_receiver_{cid}_class_{c}'
                    directory.mkdir(parents=True,exist_ok=True)
                    write_json(directory/'witness_lock.json',plan)
                    edges=[(a,b) for owner in plan['owners'] if owner!=cid for a,b in ((cid,owner),(owner,cid))]
                    transport=Transport(directory/'scope_wire.jsonl',edges,
                        Protocol(max_incoming_bytes=32*1024*1024,max_outgoing_bytes=32*1024*1024).validate())
                    try:
                        receipts=collect_current_receipts(cid,c,models[cid],routers[cid],self.config,
                            seen,plan,self.views,transport)
                        write_json(directory/'negative_receipts.json',receipts)
                        result=reg.extend_current_scope(models[cid],routers[cid],c,receipts)
                    finally:
                        wire=directory/'scope_wire.jsonl'
                        if wire.exists():
                            for line in wire.read_text(encoding='utf-8').splitlines():
                                self.communication.append(dict(task=task,round=round_id,phase=phase,**json.loads(line)))
                    result.update(receiver=cid,class_id=c,plan=plan,
                        remaining_missing_classes=sorted(set(seen)-set(validated_scope(e)['classes'])))
                    write_json(directory/'scope_update.json',result)
                    reports.append(result)
                except Rejected as exc:
                    if exc.reason=='CUMULATIVE_CAL_NEW_PROTECTION_CONFLICT':
                        e['new_conflict_latched']=True
                        reg.sync(models[cid])
                    reports.append(dict(receiver=cid,class_id=c,issued=False,reason=exc.reason,detail=exc.detail))
        return reports

    def check_lifecycle(self, task, models, routers, seen):
        scope = self.runtime_scope(seen)
        reports = []
        for cid, model in models.items():
            model.appliance_runtime_scope = copy.deepcopy(scope)
            reg = self.registry(model)
            if not reg.entries:
                continue
            before = {c: reg.head_matches(model, c) for c in reg.entries}
            reg.protect(model)
            for c in sorted(reg.entries):
                monitor = None
                if cid in self.views and self.views[cid].task == task:
                    monitor = reg.current_negative_gate(model, routers[cid], c, self.views[cid], seen,
                        str(next(model.parameters()).device), self.contract['batch_size'], runtime_scope=scope)
                    result = monitor['lifecycle']
                else:
                    result = reg.update_lifecycle(model, routers[cid], c, scope, int(task))
                reports.append(dict(receiver=cid, class_id=c, head_exact_before=before[c],
                    head_exact_after=reg.head_matches(model, c),
                    certificate_current=reg.certificate_current(model, routers[cid], c),
                    lifecycle=result, monitor=monitor))
        return reports

    def communication_summary(self):
        by_kind = {}
        for r in self.communication:
            by_kind[r['kind']] = by_kind.get(r['kind'], 0) + r['bytes']
        return dict(application_egress_bytes=sum(r['bytes'] for r in self.communication),
                    by_kind=by_kind, includes_discovery=True, includes_setup_capsules=True,
                    includes_failed_attempts=True, baseline_aggregation_bytes_separate=True,
                    limitation='simulated application payload; socket/TLS overhead unmeasured')

    def state_dict(self):
        return dict(contract=copy.deepcopy(self.contract),
            calibration_views={cid: v.state() for cid, v in self.views.items()},
            base_views={cid: v.state() for cid, v in self.base_views.items()},
            shields={cid: s.state() for cid, s in self.shields.items()},
            rounds=copy.deepcopy(self.rounds), communication=copy.deepcopy(self.communication),
            attempts=sorted(self.attempts))
