# """
# Worker node: one process's full training loop in a RingSync job.

# Each worker:
#   1. Builds the model from the SAME seed as every other worker (and
#      the baseline) -- identical initial weights everywhere is a
#      precondition for the distributed run to be comparable to anything.
#   2. Loads only its own shard of the dataset.
#   3. Each step: forward + backward locally (producing local, per-worker
#      gradients) -> flatten gradients -> ring_all_reduce across all
#      workers (producing the averaged gradient, identical on every
#      worker) -> unflatten back into .grad -> optimizer.step().

# Because every worker applies the identical averaged gradient to an
# identically-initialized model, model weights stay bit-for-bit identical
# across all workers throughout training -- this is the invariant the
# correctness test in scripts/run_benchmark.py checks for.
# """

# import argparse
# import json
# import sys
# import time
# from pathlib import Path

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# from ringsync.model import build_model
# from ringsync.data import get_worker_dataloader
# from ringsync.network.topology import setup_ring
# from ringsync.network.protocol import send_message, recv_message, MsgType, tensor_to_bytes, bytes_to_tensor
# from ringsync.allreduce.ring_allreduce import ring_all_reduce
# from ringsync.worker.grad_utils import flatten_grads, unflatten_and_set_grads


# def make_send_fn(sock):
#     def send_fn(arr: np.ndarray) -> None:
#         send_message(sock, MsgType.TENSOR_CHUNK, tensor_to_bytes(arr))
#     return send_fn


# def make_recv_fn(sock):
#     def recv_fn() -> np.ndarray:
#         msg_type, payload = recv_message(sock)
#         assert msg_type == MsgType.TENSOR_CHUNK, f"unexpected message type {msg_type}"
#         num_floats = len(payload) // 4
#         return bytes_to_tensor(payload, (num_floats,))
#     return recv_fn


# def run_worker(
#     rank: int,
#     world_size: int,
#     worker_addresses: "list[tuple[str, int]]",
#     epochs: int = 5,
#     batch_size: int = 32,
#     lr: float = 0.01,
#     seed: int = 42,
#     use_synthetic: bool = True,
#     shuffle: bool = False,
#     results_dir: "Path | None" = None,
#     num_samples: int = 2000,
# ):
#     # Without this, every worker process independently tries to use ALL
#     # available CPU cores for its own internal matmul/conv ops (PyTorch's
#     # default CPU threading behavior). Run 8 worker processes on a
#     # 16-core machine without this line and you get 8 processes each
#     # spinning up ~16 threads -- 128 threads fighting over 16 real
#     # cores. That oversubscription actively makes compute SLOWER as you
#     # add more workers, which defeats the entire point of parallelizing.
#     # Pinning each process to 1 thread and relying on the OS scheduler
#     # to place separate PROCESSES on separate cores is the correct model
#     # for CPU-based data parallelism (this is also what real
#     # multi-process CPU training setups, e.g. torch.distributed with
#     # gloo backend, recommend doing).
#     torch.set_num_threads(1)

#     model = build_model(seed=seed)  # identical init across every worker
#     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
#     criterion = nn.CrossEntropyLoss()

#     loader = get_worker_dataloader(
#         rank=rank, world_size=world_size, batch_size=batch_size,
#         use_synthetic=use_synthetic, seed=seed, num_samples=num_samples,
#     )
#     if not shuffle:
#         # deterministic, non-shuffled shard traversal -- makes each
#         # worker's exact batch sequence reproducible run to run, which
#         # matters for the correctness checks in the benchmark script.
#         loader = torch.utils.data.DataLoader(
#             loader.dataset, batch_size=batch_size, shuffle=False, drop_last=True,
#         )

#     print(f"[rank {rank}] connecting ring...")
#     inbound_sock, outbound_sock = setup_ring(rank, world_size, worker_addresses)
#     print(f"[rank {rank}] ring connected.")

#     send_fn = make_send_fn(outbound_sock)
#     recv_fn = make_recv_fn(inbound_sock)

#     loss_history = []
#     compute_time_total = 0.0
#     comm_time_total = 0.0
#     start = time.time()

#     for epoch in range(epochs):
#         epoch_losses = []
#         for step, (x, y) in enumerate(loader):
#             t0 = time.time()
#             optimizer.zero_grad()
#             out = model(x)
#             loss = criterion(out, y)
#             loss.backward()
#             t1 = time.time()  # end of local compute

#             flat_grad, layout = flatten_grads(model)
#             averaged_grad = ring_all_reduce(
#                 flat_grad, rank, world_size, send_fn, recv_fn, average=True,
#             )
#             unflatten_and_set_grads(model, averaged_grad, layout)
#             t2 = time.time()  # end of communication

#             optimizer.step()

#             compute_time_total += (t1 - t0)
#             comm_time_total += (t2 - t1)

#             epoch_losses.append(loss.item())
#             loss_history.append({
#                 "epoch": epoch, "step": step, "loss": loss.item(),
#                 "compute_s": t1 - t0, "comm_s": t2 - t1,
#             })

#         avg = sum(epoch_losses) / len(epoch_losses)
#         print(f"[rank {rank}] epoch {epoch+1}/{epochs}  avg_loss={avg:.4f}")

#     elapsed = time.time() - start
#     print(
#         f"[rank {rank}] done. wall_clock={elapsed:.2f}s "
#         f"compute={compute_time_total:.2f}s comm={comm_time_total:.2f}s"
#     )

#     inbound_sock.close()
#     outbound_sock.close()

#     if results_dir is not None:
#         results_dir = Path(results_dir)
#         results_dir.mkdir(exist_ok=True, parents=True)
#         out_path = results_dir / f"worker_{rank}_history.json"
#         with open(out_path, "w") as f:
#             json.dump({
#                 "rank": rank,
#                 "loss_history": loss_history,
#                 "wall_clock_seconds": elapsed,
#                 "compute_seconds": compute_time_total,
#                 "comm_seconds": comm_time_total,
#                 "final_state_dict": {k: v.tolist() for k, v in model.state_dict().items()},
#             }, f)
#         print(f"[rank {rank}] saved -> {out_path}")

#     return model, loss_history


# def parse_addresses(raw: str) -> "list[tuple[str, int]]":
#     """Parse 'host1:port1,host2:port2,...' into a list of (host, port)."""
#     addrs = []
#     for entry in raw.split(","):
#         host, port = entry.rsplit(":", 1)
#         addrs.append((host, int(port)))
#     return addrs


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="RingSync worker node")
#     parser.add_argument("--rank", type=int, required=True)
#     parser.add_argument("--world-size", type=int, required=True)
#     parser.add_argument("--addresses", type=str, required=True,
#                          help="comma-separated host:port list, one per rank")
#     parser.add_argument("--epochs", type=int, default=5)
#     parser.add_argument("--batch-size", type=int, default=32)
#     parser.add_argument("--lr", type=float, default=0.01)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--results-dir", type=str, default=None)
#     args = parser.parse_args()

#     run_worker(
#         rank=args.rank,
#         world_size=args.world_size,
#         worker_addresses=parse_addresses(args.addresses),
#         epochs=args.epochs,
#         batch_size=args.batch_size,
#         lr=args.lr,
#         seed=args.seed,
#         results_dir=args.results_dir,
#     )


"""
Worker node: one process's full training loop in a RingSync job.

Each worker:
  1. Builds the model from the SAME seed as every other worker (and
     the baseline) -- identical initial weights everywhere is a
     precondition for the distributed run to be comparable to anything.
  2. Loads only its own shard of the dataset.
  3. Each step: forward + backward locally (producing local, per-worker
     gradients) -> flatten gradients -> ring_all_reduce across all
     workers (producing the averaged gradient, identical on every
     worker) -> unflatten back into .grad -> optimizer.step().

Because every worker applies the identical averaged gradient to an
identically-initialized model, model weights stay bit-for-bit identical
across all workers throughout training -- this is the invariant the
correctness test in scripts/run_benchmark.py checks for.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ringsync.model import build_model
from ringsync.data import get_worker_dataloader
from ringsync.network.topology import setup_ring
from ringsync.network.protocol import send_message, recv_message, MsgType, tensor_to_bytes, bytes_to_tensor
from ringsync.allreduce.ring_allreduce import ring_all_reduce
from ringsync.worker.grad_utils import flatten_grads, unflatten_and_set_grads


def make_send_fn(sock):
    def send_fn(arr: np.ndarray) -> None:
        send_message(sock, MsgType.TENSOR_CHUNK, tensor_to_bytes(arr))
    return send_fn


def make_recv_fn(sock):
    def recv_fn() -> np.ndarray:
        msg_type, payload = recv_message(sock)
        assert msg_type == MsgType.TENSOR_CHUNK, f"unexpected message type {msg_type}"
        num_floats = len(payload) // 4
        return bytes_to_tensor(payload, (num_floats,))
    return recv_fn


def run_worker(
    rank: int,
    world_size: int,
    worker_addresses: "list[tuple[str, int]]",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.01,
    seed: int = 42,
    use_synthetic: bool = True,
    shuffle: bool = False,
    results_dir: "Path | None" = None,
    num_samples: int = 2000,
):
    # Without this, every worker process independently tries to use ALL
    # available CPU cores for its own internal matmul/conv ops (PyTorch's
    # default CPU threading behavior). Run 8 worker processes on a
    # 16-core machine without this line and you get 8 processes each
    # spinning up ~16 threads -- 128 threads fighting over 16 real
    # cores. That oversubscription actively makes compute SLOWER as you
    # add more workers, which defeats the entire point of parallelizing.
    # Pinning each process to 1 thread and relying on the OS scheduler
    # to place separate PROCESSES on separate cores is the correct model
    # for CPU-based data parallelism (this is also what real
    # multi-process CPU training setups, e.g. torch.distributed with
    # gloo backend, recommend doing).
    torch.set_num_threads(1)

    model = build_model(seed=seed)  # identical init across every worker
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    loader = get_worker_dataloader(
        rank=rank, world_size=world_size, batch_size=batch_size,
        use_synthetic=use_synthetic, seed=seed, num_samples=num_samples,
    )
    if not shuffle:
        # deterministic, non-shuffled shard traversal -- makes each
        # worker's exact batch sequence reproducible run to run, which
        # matters for the correctness checks in the benchmark script.
        loader = torch.utils.data.DataLoader(
            loader.dataset, batch_size=batch_size, shuffle=False, drop_last=True,
        )

    print(f"[rank {rank}] connecting ring...")
    inbound_sock, outbound_sock = setup_ring(rank, world_size, worker_addresses)
    print(f"[rank {rank}] ring connected.")

    send_fn = make_send_fn(outbound_sock)
    recv_fn = make_recv_fn(inbound_sock)

    loss_history = []
    compute_time_total = 0.0
    comm_time_total = 0.0
    start = time.time()

    for epoch in range(epochs):
        epoch_losses = []
        for step, (x, y) in enumerate(loader):
            t0 = time.time()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            t1 = time.time()  # end of local compute

            flat_grad, layout = flatten_grads(model)
            averaged_grad = ring_all_reduce(
                flat_grad, rank, world_size, send_fn, recv_fn, average=True,
            )
            unflatten_and_set_grads(model, averaged_grad, layout)
            t2 = time.time()  # end of communication

            optimizer.step()

            compute_time_total += (t1 - t0)
            comm_time_total += (t2 - t1)

            epoch_losses.append(loss.item())
            loss_history.append({
                "epoch": epoch, "step": step, "loss": loss.item(),
                "compute_s": t1 - t0, "comm_s": t2 - t1,
            })

        avg = sum(epoch_losses) / len(epoch_losses)
        print(f"[rank {rank}] epoch {epoch+1}/{epochs}  avg_loss={avg:.4f}")

        # Machine-parseable progress sentinel -- rank 0 only, since all
        # workers in synchronous training progress in lockstep (each
        # step is a barrier), so rank 0's epoch count is a fair proxy
        # for the whole config's progress without every worker
        # printing redundant/interleaved sentinels for the same epoch.
        if rank == 0:
            print(f'RS_PROGRESS|{{"type":"epoch","config":"world_size_{world_size}",'
                  f'"epoch":{epoch+1},"total_epochs":{epochs}}}')

    elapsed = time.time() - start
    print(
        f"[rank {rank}] done. wall_clock={elapsed:.2f}s "
        f"compute={compute_time_total:.2f}s comm={comm_time_total:.2f}s"
    )

    inbound_sock.close()
    outbound_sock.close()

    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(exist_ok=True, parents=True)
        out_path = results_dir / f"worker_{rank}_history.json"
        with open(out_path, "w") as f:
            json.dump({
                "rank": rank,
                "loss_history": loss_history,
                "wall_clock_seconds": elapsed,
                "compute_seconds": compute_time_total,
                "comm_seconds": comm_time_total,
                "final_state_dict": {k: v.tolist() for k, v in model.state_dict().items()},
            }, f)
        print(f"[rank {rank}] saved -> {out_path}")

    return model, loss_history


def parse_addresses(raw: str) -> "list[tuple[str, int]]":
    """Parse 'host1:port1,host2:port2,...' into a list of (host, port)."""
    addrs = []
    for entry in raw.split(","):
        host, port = entry.rsplit(":", 1)
        addrs.append((host, int(port)))
    return addrs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RingSync worker node")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--addresses", type=str, required=True,
                         help="comma-separated host:port list, one per rank")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()

    run_worker(
        rank=args.rank,
        world_size=args.world_size,
        worker_addresses=parse_addresses(args.addresses),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        results_dir=args.results_dir,
    )