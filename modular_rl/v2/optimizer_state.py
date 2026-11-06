"""Initialize lazy offloaded Adam state without applying an optimizer update."""


def initialize_hybrid_state(hybrid):
    import torch

    if not hasattr(hybrid, 'sub_optimizers'):
        return
    for inner in hybrid.sub_optimizers:
        for group in inner.param_groups:
            if isinstance(inner, (torch.optim.Adam, torch.optim.AdamW)):
                saved = [(p, p.grad) for p in group['params']]
                try:
                    for p, _ in saved:
                        # _init_group is the same lazy initialization used by step(),
                        # but performs neither an update nor a clock increment.
                        if not inner.state[p]:
                            p.grad = torch.zeros_like(p)
                    inner._init_group(group, *[[] for _ in range(6)])
                finally:
                    for p, grad in saved:
                        p.grad = grad
            else:
                for p in group['params']:
                    if not inner.state[p]:
                        inner.initialize_state(p)
    hybrid._sync_sub_optimizers_state_to_hdo()
