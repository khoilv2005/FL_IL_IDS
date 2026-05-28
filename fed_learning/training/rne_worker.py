"""Multi-GPU worker for RNE clients."""

from collections import OrderedDict
from typing import Dict

from ..models.rne_model import RNEModel
from .der_worker import DERWorker


class RNEWorker(DERWorker):
    """Worker for RNE with dynamic model reconstruction."""

    def create_model(self):
        return RNEModel(
            self.config["input_shape"],
            self.config["num_classes"],
            recurrent_scale=self.config.get("rne_recurrent_scale", 1.0),
        ).to(self.device)

    def prepare_model(self, model):
        _reconstruct_model_structure(model, self.global_params, self.config)
        model.to(self.device)

    def run(self):
        import time
        import torch

        gpu_start = time.time()
        model = self.create_model()
        self.prepare_model(model)

        for idx, client in enumerate(self.clients):
            init_params = self.get_init_params(client)
            self.load_params(model, init_params)
            client.setup_for_gpu(model, self.device)
            self.prepare_client(client, model, idx)

            result = client.train(
                trainer=self.trainer,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                **self.get_train_kwargs(client, idx),
            )
            self.results_dict[client.client_id] = result

        gpu_time = time.time() - gpu_start
        print(
            f"      [{self.device_name}] RNE Stage {self.stage}: "
            f"{len(self.clients)} clients done in {gpu_time:.1f}s"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train_rne_clients_on_gpu(
    gpu_id: int,
    clients: list,
    global_params: OrderedDict,
    config: Dict,
    results_dict: Dict,
    trainer,
    use_cpu: bool = False,
    stage: int = 1,
):
    worker = RNEWorker(
        gpu_id,
        clients,
        global_params,
        config,
        results_dict,
        trainer,
        use_cpu,
        stage,
    )
    worker.run()


def _reconstruct_model_structure(
    model: RNEModel,
    global_params: OrderedDict,
    config: Dict,
):
    extractor_indices = set()
    for key in global_params.keys():
        if key.startswith("extractors."):
            parts = key.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                extractor_indices.add(int(parts[1]))

    num_tasks = len(extractor_indices)
    if num_tasks == 0:
        return

    task_classes_history = config.get("task_classes_history", {})
    s_max = config.get("s_max", 15.0)
    for task_idx in range(num_tasks):
        if task_idx in task_classes_history:
            new_classes = task_classes_history[task_idx]
        elif str(task_idx) in task_classes_history:
            new_classes = task_classes_history[str(task_idx)]
        else:
            head_key = f"classifier_heads.{task_idx}.weight"
            if head_key in global_params:
                new_classes = list(range(global_params[head_key].shape[0]))
            else:
                new_classes = [0]
        model.add_task(new_classes, s_max=s_max)
