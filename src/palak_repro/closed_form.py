"""UCE's regularized closed-form projection edit and the EEB shallow mapping."""
import torch

from .model import attention_projections, concept_vector


@torch.no_grad()
def edit_projections(pipe, mappings, preserve=(), lamb=0.5, edit_scale=1.0, preserve_scale=1.0):
    """Map input phrases to teacher outputs for output phrases in each attn2 K/V matrix."""
    if lamb <= 0:
        raise ValueError("UCE lambda must be positive")
    phrases = set(preserve) | {x for pair in mappings for x in pair}
    vectors = {p: concept_vector(pipe, p) for p in phrases}
    modules = attention_projections(pipe.unet)
    if not modules:
        raise RuntimeError("No cross-attention key/value projections found")
    for _, module in modules:
        # MPS lacks linalg.solve; the edit is small enough to solve on CPU.
        solve_device = torch.device("cpu") if module.weight.device.type == "mps" else module.weight.device
        old = module.weight.detach().to(device=solve_device, dtype=torch.float32).clone()
        dim = old.shape[1]
        lhs = lamb * torch.eye(dim, device=old.device)
        rhs = lamb * old
        for source, destination in mappings:
            c = vectors[source].to(solve_device).reshape(-1, 1)
            d = vectors[destination].to(solve_device).reshape(-1, 1)
            lhs.add_(edit_scale * (c @ c.T))
            rhs.add_(edit_scale * ((old @ d) @ c.T))
        for phrase in preserve:
            c = vectors[phrase].to(solve_device).reshape(-1, 1)
            lhs.add_(preserve_scale * (c @ c.T))
            rhs.add_(preserve_scale * ((old @ c) @ c.T))
        # Solve W @ lhs = rhs without an explicit matrix inverse.
        module.weight.copy_(torch.linalg.solve(lhs, rhs.T).T.to(module.weight))
    return len(modules)


def inject_shallow(pipe, target, trigger, cfg):
    # EEB appendix B: simultaneous target <- trigger and anchor <- target.
    return edit_projections(pipe, [(trigger, target), (target, "")],
                            lamb=cfg["lambda"], edit_scale=cfg["edit_scale"],
                            preserve_scale=cfg["preserve_scale"])


def erase_uce(pipe, target, cfg, preserve=()):
    return edit_projections(pipe, [(target, cfg["anchor"])], preserve=preserve,
                            lamb=cfg["lambda"], edit_scale=cfg["edit_scale"],
                            preserve_scale=cfg["preserve_scale"])
