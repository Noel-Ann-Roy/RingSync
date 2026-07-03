"""
Gradient flatten/unflatten utilities.

ring_all_reduce operates on a single flat numpy array. A model's
gradients live as N separate tensors (one per parameter, all different
shapes -- conv kernels, biases, linear weights). This module bridges
the two: flatten every parameter's .grad into one contiguous buffer
before the ring exchange, then scatter the reduced result back into
each parameter's .grad afterward.

This mirrors what real DDP implementations call "gradient bucketing" --
flattening avoids doing a separate, small network message per
parameter (545,098 params in our model, but only a handful of tensors
-- flattening means one ring exchange per step, not one per tensor).

Critically: every worker must flatten parameters in the EXACT SAME
ORDER for this to produce correct results, since the ring exchange has
no idea what a "conv1.weight" is -- it's just summing byte offsets.
`model.named_parameters()` iterates in registration order, which is
deterministic and identical across processes as long as every worker
constructs the model architecture identically (same class, same seed
for initialization doesn't even matter here -- only structure/order
matters for this part).
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


def flatten_grads(model: nn.Module) -> Tuple[np.ndarray, List[Tuple[str, tuple, int, int]]]:
    """
    Returns:
      flat: a single 1D float32 numpy array containing every
            parameter's gradient, concatenated in named_parameters() order.
      layout: list of (name, shape, start_offset, end_offset) describing
              where each parameter's slice lives in `flat`, so
              unflatten_grads can put things back correctly.
    """
    pieces = []
    layout = []
    offset = 0

    for name, param in model.named_parameters():
        if param.grad is None:
            # A parameter with no gradient this step (shouldn't normally
            # happen for a densely-connected small CNN, but guard anyway)
            grad_np = np.zeros(param.numel(), dtype=np.float32)
        else:
            grad_np = param.grad.detach().cpu().numpy().flatten().astype(np.float32)

        pieces.append(grad_np)
        shape = tuple(param.shape)
        layout.append((name, shape, offset, offset + grad_np.size))
        offset += grad_np.size

    flat = np.concatenate(pieces) if pieces else np.array([], dtype=np.float32)
    return flat, layout


def unflatten_and_set_grads(
    model: nn.Module, flat: np.ndarray, layout: List[Tuple[str, tuple, int, int]]
) -> None:
    """
    Inverse of flatten_grads: slices `flat` back into per-parameter
    chunks (using the same layout produced during flattening) and
    assigns them as each parameter's .grad, ready for optimizer.step().
    """
    params = dict(model.named_parameters())
    for name, shape, start, end in layout:
        chunk = flat[start:end].reshape(shape)
        grad_tensor = torch.from_numpy(chunk.copy())
        param = params[name]
        if param.grad is None:
            param.grad = grad_tensor.clone()
        else:
            param.grad.copy_(grad_tensor)


if __name__ == "__main__":
    import torch.nn.functional as F

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 3)
            self.b = nn.Linear(3, 2)

        def forward(self, x):
            return self.b(F.relu(self.a(x)))

    torch.manual_seed(0)
    model = Tiny()
    x = torch.randn(5, 4)
    y = torch.randint(0, 2, (5,))
    loss = F.cross_entropy(model(x), y)
    loss.backward()

    original_grads = {name: p.grad.clone() for name, p in model.named_parameters()}

    flat, layout = flatten_grads(model)
    print(f"Flattened gradient length: {flat.shape}")

    # zero out grads, then reconstruct from flat -- should get identical values back
    for p in model.parameters():
        p.grad.zero_()

    unflatten_and_set_grads(model, flat, layout)

    for name, p in model.named_parameters():
        assert torch.allclose(p.grad, original_grads[name]), f"mismatch in {name}"

    print("flatten/unflatten round-trip OK -- all parameter grads reconstructed exactly")
