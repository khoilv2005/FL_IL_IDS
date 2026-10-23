"""Receiver-owned DRAFT experts. No native module registration or deployment API."""
from collections import Counter
from copy import deepcopy
import hashlib
import pickle

import numpy as np
import torch

VERSION = 'candle_v4_native_shadow_v1'


def digest(value):
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().contiguous()
            h.update(str((str(x.dtype), tuple(x.shape))).encode())
            h.update(x.numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((str(x.dtype), x.shape)).encode()); h.update(x.tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=lambda k: (str(type(k)), str(k))):
                visit(key); visit(x[key])
        elif isinstance(x, (list, tuple)):
            for item in x: visit(item)
        else:
            h.update(repr(x).encode())
        h.update(b'\0')
    visit(value)
    return h.hexdigest()


def native_fingerprint(models, detectors):
    from fed_learning.training.checkpoint_state import snapshot_denice_state
    return digest({cid: (model.state_dict(), snapshot_denice_state(model, detectors[cid]),
                        [(name, module.training) for name,module in model.named_modules()])
                   for cid, model in models.items()})


def make_expert(seed):
    # Constructors consume only a forked CPU RNG. CUDA initialization is not used.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return torch.nn.Sequential(torch.nn.Linear(39, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))


class ShadowService:
    def __init__(self, spec, saved=None):
        self.spec = deepcopy(spec)
        required = {'task', 'receiver', 'donor', 'target', 'current_classes', 'input_sha256', 'fit_paths'}
        if not required <= set(spec) or spec.get('mode') != 'DRAFT':
            raise ValueError('Explicit DRAFT current-task contract required')
        if spec['target'] not in spec['current_classes'] or spec['receiver'] == spec['donor']:
            raise ValueError('Invalid current target or peer')
        self.branches = {}
        for name in spec.get('branch_names', ('donor_only', 'joint_class_balanced', 'joint_unbalanced')):
            model = make_expert(42)
            self.branches[name] = (model, torch.optim.Adam(model.parameters(), lr=.001))
        self.steps = 0; self.pending = 0; self.last_round = -1
        self.bytes = 0; self.messages = 0; self.by_kind = Counter(); self.history = []
        self.expired = False
        if saved:
            if saved['version'] != VERSION or saved['spec'] != self.spec:
                raise ValueError('Shadow continuation protocol changed')
            for name, (model, optimizer) in self.branches.items():
                model.load_state_dict(saved['branches'][name]['model'])
                optimizer.load_state_dict(saved['branches'][name]['optimizer'])
            for key in ('steps', 'pending', 'last_round', 'bytes', 'messages', 'history', 'expired'):
                setattr(self, key, deepcopy(saved[key]))
            self.by_kind = Counter(saved['by_kind'])

    def state_dict(self):
        return deepcopy(dict(version=VERSION, spec=self.spec,
            branches={name: dict(model=m.state_dict(), optimizer=o.state_dict())
                      for name, (m, o) in self.branches.items()},
            steps=self.steps, pending=self.pending, last_round=self.last_round,
            bytes=self.bytes, messages=self.messages, by_kind=dict(self.by_kind),
            history=self.history, expired=self.expired))

    def send(self, kind, message):
        raw = pickle.dumps(message, protocol=5)
        if self.bytes + len(raw) > self.spec.get('wire_budget', 128 * 1024**2):
            raise RuntimeError('Shadow wire budget exhausted; no partial native commit')
        self.bytes += len(raw); self.messages += 1; self.by_kind[kind] += len(raw)
        return pickle.loads(raw)

    def data(self, client):
        # This view is already current-task, post-native-local-validation BASE.
        x = client.X_train.detach().cpu(); y = client.y_train.detach().cpu()
        if x.shape[1:] != (39,) or not set(y.tolist()) <= set(self.spec['current_classes']):
            raise ValueError('Off-task expert BASE view')
        indices = []
        for c in sorted(set(y.tolist())):
            seen = set(); chosen = []
            for i in torch.nonzero(y == c).flatten().tolist():
                key = hashlib.sha256(x[i].numpy().tobytes()).digest()
                if key not in seen: chosen.append(i); seen.add(key)
                if len(chosen) == 4096: break
            indices.extend(chosen)
        index = torch.tensor(indices, dtype=torch.long)
        return x[index], y[index]

    def train_step(self, name, data, step, task, round_id):
        target = self.spec['target']; donor = self.spec['donor']; receiver = self.spec['receiver']
        owners = [donor] if name == 'donor_only' else sorted([donor, receiver])
        counts = {o: Counter(data[o][1].tolist()) for o in owners}
        negative = [(o, c, n) for o in owners for c, n in sorted(counts[o].items()) if c != target]
        nneg = sum(n for _, _, n in negative)
        by_class = Counter()
        for _, c, n in negative: by_class[c] += n
        generator = torch.Generator().manual_seed(42 + step * 100003 + 202)
        tags = torch.randint(nneg, (128,), generator=generator)
        allocation = {o: Counter() for o in owners}; allocation[donor][target] = 128
        offset = 0
        for o, c, n in negative:
            allocation[o][c] = int(((tags >= offset) & (tags < offset + n)).sum()); offset += n
        model, optimizer = self.branches[name]
        reference = deepcopy(model.state_dict()); reference_sha = digest(reference)
        identity = dict(task=task, round=round_id, step=step, target=target,
                        receiver=receiver, donor=donor, version=VERSION,
                        input_sha256=self.spec['input_sha256'], reference_sha256=reference_sha,
                        layout='39-64-1-fp32', mask='all-expert-parameters-young-v1')
        gradient = [torch.zeros_like(p) for p in model.parameters()]
        loss_sum = 0.; receiver_norm = 0.; positive_count = 0
        for owner in owners:
            requests = {c: n for c, n in allocation[owner].items() if n}
            if not requests: continue
            peer = make_expert(42)
            if name != 'donor_only' and owner == donor:
                packet = self.send(name + '/reference', dict(identity=identity, state=reference,
                                                           requests=requests, counts=counts))
                if packet['identity'] != identity or digest(packet['state']) != reference_sha:
                    raise ValueError('Stale shadow reference')
                peer.load_state_dict(packet['state'])
            else: peer.load_state_dict(reference)
            rows = []; weights = []
            x, y = data[owner]
            for c, n in sorted(requests.items()):
                g = torch.Generator().manual_seed(42 + step * 100003 + (0 if c == target else owner * 1009) + c * 17)
                local = torch.nonzero(y == c).flatten()
                rows.append(local[torch.randint(len(local), (n,), generator=g)])
                w = .5 / 128
                if c != target and name == 'joint_class_balanced': w *= nneg / (len(by_class) * by_class[c])
                weights.extend([w] * n)
            idx = torch.cat(rows); truth = (y[idx] == target).float()
            positive_count += int(truth.sum())
            if owner == receiver and truth.any(): raise ValueError('Receiver target data forbidden')
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(peer(x[idx]).flatten(), truth,
                    reduction='none') * torch.tensor(weights)).sum()
            grads = [g.detach() for g in torch.autograd.grad(loss, list(peer.parameters()))]
            if not all(torch.isfinite(g).all() for g in grads): raise FloatingPointError('Nonfinite expert gradient')
            if name != 'donor_only' and owner == donor:
                packet = self.send(name + '/gradient', dict(identity=identity, gradients=grads))
                if packet['identity'] != identity: raise ValueError('Stale gradient')
                grads = packet['gradients']
            for total, value in zip(gradient, grads): total.add_(value)
            loss_sum += float(loss.detach())
            if owner == receiver: receiver_norm = float(sum(g.square().sum() for g in grads).sqrt())
        if positive_count != 128: raise ValueError('Missing donor positive contribution')
        optimizer.zero_grad(set_to_none=True)
        for p, g in zip(model.parameters(), gradient): p.grad = g
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()): raise FloatingPointError('Nonfinite expert parameters')
        return dict(loss=loss_sum, receiver_negative_gradient_norm=receiver_norm)

    def after_round(self, task, round_id, models, detectors, clients, active_ids, graph):
        if task != self.spec['task']:
            self.expired = True; self.pending = 0
            return dict(status='EXPIRED', task=task, steps=self.steps)
        if self.expired: raise ValueError('Expired expert generation cannot resume training')
        if round_id != self.last_round + 1: raise ValueError('Shadow round repeated or skipped')
        before = native_fingerprint(models, detectors)
        if not all(torch.isfinite(p).all() for model in models.values() for p in model.parameters()):
            raise FloatingPointError('Nonfinite native parameters before shadow')
        self.pending += 44
        donor, receiver = self.spec['donor'], self.spec['receiver']
        row = graph.get('alpha_debug', {}).get(receiver, graph.get('alpha_debug', {}).get(str(receiver), {}))
        ids, alphas = row.get('group_ids', []), row.get('alphas', [])
        if len(ids) != len(alphas): raise ValueError('Malformed current native graph')
        eligible = any(int(j) == donor and np.isfinite(a) and a > 0 for j, a in zip(ids, alphas))
        eligible &= donor in active_ids and receiver in active_ids
        start_bytes = self.bytes; updates = 0; details = {}
        if eligible:
            data = {o: self.data(clients[o]) for o in (donor, receiver)}
            if (data[donor][1] == self.spec['target']).sum() == 0 or (data[receiver][1] == self.spec['target']).any():
                raise ValueError('Invalid donor ownership / receiver functional fixture')
            if any(not (y != self.spec['target']).any() for _, y in data.values()):
                raise ValueError('Missing current negatives')
            updates = min(88, self.pending)
            for _ in range(updates):
                for name in self.branches:
                    details[name] = self.train_step(name, data, self.steps, task, round_id)
                self.steps += 1; self.pending -= 1
            metrics = self.evaluate(models[receiver], detectors[receiver], data, task, round_id)
        else: metrics = None
        if before != native_fingerprint(models, detectors): raise RuntimeError('Shadow mutated native state')
        self.last_round = round_id
        receipt = dict(task=task, round=round_id, status='DRAFT', eligible=bool(eligible), updates=updates,
                       total_steps=self.steps, pending=self.pending, native_sha256=before,
                       round_wire_bytes=self.bytes-start_bytes, total_wire_bytes=self.bytes,
                       gradients=details, FIT=metrics)
        self.history.append(receipt)
        print(f'CANDLE V4 SHADOW T{task} R{round_id} eligible={bool(eligible)} updates={updates} total={self.steps} bytes={self.bytes}', flush=True)
        return receipt

    @torch.no_grad()
    def evaluate(self, base, detector, train, task, round_id):
        from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes
        from fed_learning.training.checkpoint_state import snapshot_denice_state
        donor, receiver = self.spec['donor'], self.spec['receiver']; target = self.spec['target']
        fit = {}
        train_hashes = {hashlib.sha256(row.numpy().tobytes()).digest() for x, _ in train.values() for row in x}
        for owner in (donor, receiver):
            with np.load(self.spec['fit_paths'][str(owner)], allow_pickle=False) as z:
                x = torch.from_numpy(z['X'].copy()); y = torch.from_numpy(z['y'].copy())
            if x.shape[1:] != (39,) or not set(y.tolist()) <= set(self.spec['current_classes']):
                raise ValueError('Off-task FIT panel')
            if any(hashlib.sha256(row.numpy().tobytes()).digest() in train_hashes for row in x):
                raise ValueError('BASE/FIT content overlap')
            fit[owner] = (x, y)
        # Count the actual current receiver inference reference every eligible round.
        payload = dict(weights={k:v.detach().cpu() for k,v in base.state_dict().items()},
                       algorithm=snapshot_denice_state(base, detector))
        # This is an accounting payload; raw reference caches must never be sent.
        context = payload['algorithm'].get('context_detector', {})
        for key in ('reference_input_memory', 'reference_label_memory'):
            context.pop(key, None)
        self.send('current_base_verification_reference', payload)
        modes = [(module,module.training) for module in base.modules()]
        base.eval(); device = next(base.parameters()).device
        baseline = {}
        for owner, (x,y) in fit.items():
            predictions=[]
            for start in range(0,len(x),512):
                out,_ = _denice_routed_logits_with_episodes(base,x[start:start+512].to(device),detector,
                    list(range(max(self.spec['current_classes'])+1)),str(device),inference_policy='pred_hard')
                predictions.append(out.argmax(1).cpu())
            baseline[owner]=torch.cat(predictions)
        for module,mode in modes: module.training=mode
        result={}
        for name,(model,_) in self.branches.items():
            self.send(name+'/FIT_expert_reference', dict(task=task,round=round_id,state=model.state_dict()))
            scores={o:torch.sigmoid(model(x)).flatten() for o,(x,_) in fit.items()}
            positive=scores[donor][fit[donor][1]==target]
            tau=float(positive.sort(descending=True).values[int(np.ceil(.95*len(positive)))-1])
            rows={}
            for threshold_name,threshold in (('fixed_0p5',.5),('donor_FIT_recall95',tau)):
                pools={}
                for owner,(_,y) in fit.items():
                    pred=torch.where(scores[owner]>=threshold,target,baseline[owner]); old=baseline[owner]==y; new=pred==y
                    pools[str(owner)]=dict(rows=len(y),target_samples=int((y==target).sum()),
                        target_correct=int(((pred==target)&(y==target)).sum()),
                        expert_positive_recall=(float((scores[owner][y==target]>=threshold).float().mean()) if (y==target).any() else None),
                        baseline_correct=int(old.sum()),candidate_correct=int(new.sum()),
                        rescue=int((~old&new).sum()),break_count=int((old&~new).sum()),
                        changed_wrong=int((~old&~new&(pred!=baseline[owner])).sum()),
                        false_activations=int(((scores[owner]>=threshold)&(y!=target)).sum()),
                        per_class={str(c):dict(n=int((y==c).sum()),false_activations=int(((scores[owner]>=threshold)&(y==c)&(y!=target)).sum()))
                                   for c in sorted(set(y.tolist()))})
                rows[threshold_name]=dict(threshold=threshold,pools=pools)
            result[name]=rows
        self.send('FIT_receipt',result)
        return result
