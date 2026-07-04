"""
Single-process training baseline.

This is the ground-truth reference for everything else in RingSync.
Its loss curve is what the distributed (ring all-reduce) training run
must reproduce -- if the distributed loss curve doesn't overlay this
one, there's a bug in the gradient averaging, not in the "ML" part of
the project.

Run:
    python baseline/train_single_process.py
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ringsync.model import build_model
from ringsync.data import get_full_dataloader

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def train(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.01,
    seed: int = 42,
    use_synthetic: bool = True,
    num_samples: int = 2000,
    num_threads: "int | None" = None,
    output_filename: str = "baseline_loss_history.json",
    output_dir: "Path | None" = None,
    progress_config: str = "baseline",
):

    if num_threads is not None:
        torch.set_num_threads(num_threads)

    torch.manual_seed(seed)

    model = build_model(seed=seed)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    loader = get_full_dataloader(
        batch_size=batch_size, use_synthetic=use_synthetic, seed=seed, num_samples=num_samples,
    )

    loss_history = []
    start = time.time()

    for epoch in range(epochs):
        epoch_losses = []
        for step, (x, y) in enumerate(loader):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            loss_history.append({"epoch": epoch, "step": step, "loss": loss.item()})

        avg = sum(epoch_losses) / len(epoch_losses)
        print(f"[baseline] epoch {epoch+1}/{epochs}  avg_loss={avg:.4f}")
        print(f'RS_PROGRESS|{{"type":"epoch","config":"{progress_config}",'
              f'"epoch":{epoch+1},"total_epochs":{epochs}}}')

    elapsed = time.time() - start
    print(f"[baseline] total wall-clock time: {elapsed:.2f}s")

    out_dir = Path(output_dir) if output_dir is not None else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_filename
    with open(out_path, "w") as f:
        json.dump({"loss_history": loss_history, "wall_clock_seconds": elapsed}, f, indent=2)

    print(f"[baseline] saved loss history -> {out_path}")
    return loss_history, elapsed


if __name__ == "__main__":
    train()