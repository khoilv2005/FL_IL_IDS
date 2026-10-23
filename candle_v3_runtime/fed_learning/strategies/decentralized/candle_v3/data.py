"""Task-scoped views: only current labels/BASE/FIT/CAL reach teaching workers."""
import hashlib
import numpy as np
import torch


class CurrentTaskRoles:
    def __init__(self, roles, task, classes, base_by_owner, seed):
        self.roles, self.task = roles, int(task)
        self.classes = tuple(sorted(int(c) for c in classes))
        self.base = base_by_owner
        self.seed = int(seed)
        self.closed = False

    def get(self, owner, role, *, task, per_class):
        if self.closed or int(task) != self.task:
            raise ValueError('Historical or closed task data access is forbidden')
        if role not in ('base', 'fit', 'calibration'):
            raise ValueError('Teaching cannot open validation or TEST')
        if role == 'base':
            x,y = self.base[int(owner)]
            x,y = x.detach().cpu(),y.detach().cpu().long()
        else:
            entry = self.roles.manifest['clients'][str(owner)]
            path = self.roles.source/f'client_{owner}_train.npz'
            indices = self.roles.root/'indices'/f'client_{owner}.npz'
            # Source and indices were verified by the BASE loader for active clients.
            if int(owner) not in self.roles.verified:
                raise ValueError('Active client source must be verified by the BASE loader first')
            with np.load(indices, allow_pickle=False) as split:
                rows = split[role]
            with np.load(path, allow_pickle=False) as source:
                labels = source['y_train']
                if len(labels) != entry['source_rows']:
                    raise ValueError('Source coordinates changed')
                rows = rows[np.isin(labels[rows], self.classes)]
                # Filter role indices to current labels before exposing any examples.
                x = torch.from_numpy(np.asarray(source['X_train'][rows], dtype=np.float32))
                y = torch.from_numpy(np.asarray(labels[rows], dtype=np.int64))
        keep = torch.isin(y, torch.as_tensor(self.classes))
        x,y = x[keep],y[keep]
        rng = np.random.default_rng(self.seed + self.task*100003 + int(owner)*1009)
        selected, ids = [], set()
        for c in self.classes:
            rows = torch.nonzero(y == c).flatten().numpy()
            added = 0
            for row in rng.permutation(rows):
                identity = hashlib.sha256(x[row].contiguous().numpy().tobytes() + int(c).to_bytes(4,'little')).hexdigest()
                if identity in ids:
                    continue
                ids.add(identity); selected.append(int(row)); added += 1
                if added >= per_class:
                    break
        return x[selected], y[selected]

    def close(self):
        self.closed = True
        self.base = {}
