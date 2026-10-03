"""Client-local EWC: task-specific diagonal Fisher penalties (paper Eq. 3)."""
import torch
from .denice_replay import inference_statistics


def consolidate_ewc(model, x, y, task_id, controls, seed=0):
    state = getattr(model, 'ewc_state', {})
    completed = state.get('completed_tasks', [])
    if int(task_id) in completed:
        raise ValueError('EWC task already consolidated')
    if not len(y):
        raise ValueError('EWC requires local training examples')
    generator = torch.Generator().manual_seed(int(seed))
    idx = torch.randperm(len(y), generator=generator)[:controls['fisher_samples']]
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    fisher = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in parameters}
    device = next(model.parameters()).device
    # Each gradient is squared BEFORE averaging. Squaring a batch gradient is
    # not a diagonal Fisher estimator. Never use peer/test inputs here.
    with torch.enable_grad(), inference_statistics(model), torch.backends.cudnn.flags(enabled=False):
        for index in idx.tolist():
            logits = model(x[index:index+1].to(device)).float()
            if controls['logit_scope'] == 'seen':
                mask = torch.as_tensor(model.unit_ranks['fc2'] > 0, device=device)
                logits = logits.masked_fill(~mask, -1e4)
            label = (torch.multinomial(logits.detach().softmax(1).cpu(), 1, generator=generator).flatten().to(device)
                     if controls['fisher_labels'] == 'model' else y[index:index+1].long().to(device))
            log_probability = logits.log_softmax(1)[0, label[0]]
            gradients = torch.autograd.grad(log_probability, [p for _, p in parameters], allow_unused=True)
            for (name, _), gradient in zip(parameters, gradients):
                if gradient is not None:
                    fisher[name].add_(gradient.detach().float().square() / len(idx))
    if any(not torch.isfinite(value).all() for value in fisher.values()):
        raise ValueError('Nonfinite EWC Fisher; state not committed')
    fisher = {n: f.detach().cpu() for n, f in fisher.items()}
    anchor = {n: p.detach().cpu().clone() for n, p in parameters}
    banks = list(state.get('banks', []))
    if controls['ewc_mode'] == 'online' and banks:
        previous = banks[-1]['fisher']
        fisher = {n: f + controls['decay'] * previous.get(n, torch.zeros_like(f)) for n, f in fisher.items()}
        banks = []
    banks.append(dict(task=int(task_id), fisher=fisher, anchor=anchor))
    model.ewc_state = dict(banks=banks, completed_tasks=completed + [int(task_id)])
    return dict(samples=len(idx), estimator=controls['fisher_labels'], mode=controls['ewc_mode'],
                banks=len(banks), fisher_trace=sum(float(f.sum()) for f in fisher.values()))


def ewc_loss_factory(model, controls):
    coefficient = controls['ewc_lambda'] / 2.
    if not coefficient:
        return None
    parameters = dict(model.named_parameters())
    terms = []
    for bank in getattr(model, 'ewc_state', {}).get('banks', []):
        for name, importance in bank['fisher'].items():
            if name not in parameters or parameters[name].shape != importance.shape:
                raise ValueError('EWC parameter topology changed: '+name)
            p = parameters[name]
            if torch.count_nonzero(importance):
                terms.append((p, bank['anchor'][name].to(p.device), importance.to(p.device)))
    if not terms:
        return None
    def loss():
        return coefficient * sum((f * (p.float() - anchor).square()).sum() for p, anchor, f in terms)
    return loss
