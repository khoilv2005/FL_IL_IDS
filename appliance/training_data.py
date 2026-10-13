"""Clean-role loader whose APPLIANCE BASE runtime opens current partitions only."""
import torch
from fed_learning.data.denice_clean_roles import CleanRoleIncrementalDataLoader
from .current_base_data import CurrentBaseData


class ApplianceIncrementalDataLoader(CleanRoleIncrementalDataLoader):
    def __init__(self, *args, base_store, role_sha256, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_store, self.role_sha256 = base_store, role_sha256
        self.current_base_views = {}

    def get_client_data(self, cid, task_id):
        if cid not in self.current_base_views:
            self.current_base_views[cid] = CurrentBaseData(self.base_store, int(cid), int(task_id), self.role_sha256)
        view = self.current_base_views[cid]
        while view.task < task_id:
            view.advance(view.task + 1)
        if view.task != task_id:
            raise ValueError('Training BASE authority cannot regress to a previous task')
        pool = view.current_pool(int(cid), 'base', self.get_task_classes(task_id))
        return torch.from_numpy(pool['X']), torch.from_numpy(pool['y'])
