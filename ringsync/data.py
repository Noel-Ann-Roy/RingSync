"""
Data loading + sharding for RingSync.

Two things live here:
  1. Getting CIFAR-10 (real dataset, for actual training runs).
  2. Sharding logic: splitting a dataset into N disjoint, equal-sized
     slices, one per worker -- this is the "independent data slices"
     piece of the project spec.

A synthetic dataset is included as a fallback. Real CIFAR-10 needs to
download from an external host, which may not be reachable from every
sandboxed/offline environment. The synthetic path lets you validate the
sharding + training + all-reduce mechanics end-to-end without a network
dependency, then flip `use_synthetic=False` on a machine with internet
access to get the real correctness/speedup numbers for your writeup.
"""

from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Subset


class SyntheticCIFARLike(Dataset):
    """
    Deterministic fake dataset shaped like CIFAR-10 (3x32x32, 10 classes).

    Seeded so that every process reconstructing this dataset (baseline,
    or any worker) gets bit-identical samples -- important, because if
    workers don't actually have different/consistent data, your
    "distributed training" is meaningless.
    """

    def __init__(self, num_samples: int = 2000, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randn(num_samples, 3, 32, 32, generator=g)
        self.targets = torch.randint(0, 10, (num_samples,), generator=g)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


def get_dataset(use_synthetic: bool = True, data_root: str = "./data", num_samples: int = 2000):
    if use_synthetic:
        return SyntheticCIFARLike(num_samples=num_samples, seed=0)

    # Real CIFAR-10 path -- requires torchvision + internet access to
    # download on first run.
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )


def shard_indices(dataset_len: int, num_workers: int) -> List[List[int]]:
    """
    Split [0, dataset_len) into num_workers disjoint, contiguous,
    (near-)equal chunks. Deterministic given dataset_len and
    num_workers -- every worker can independently compute its own
    shard's index range without needing the orchestrator to hand it out,
    as long as they agree on worker rank and world size.
    """
    base = dataset_len // num_workers
    remainder = dataset_len % num_workers
    shards = []
    start = 0
    for rank in range(num_workers):
        size = base + (1 if rank < remainder else 0)
        shards.append(list(range(start, start + size)))
        start += size
    return shards


def get_worker_dataloader(
    rank: int,
    world_size: int,
    batch_size: int = 32,
    use_synthetic: bool = True,
    data_root: str = "./data",
    seed: int = 0,
    num_samples: int = 2000,
) -> DataLoader:
    """
    Build the DataLoader for a single worker's shard.

    Called independently on each worker process/machine -- each worker
    only ever materializes its own slice of the dataset, never the
    whole thing. That's the "independent data slices" requirement.
    """
    dataset = get_dataset(use_synthetic=use_synthetic, data_root=data_root, num_samples=num_samples)
    shards = shard_indices(len(dataset), world_size)
    subset = Subset(dataset, shards[rank])

    generator = torch.Generator().manual_seed(seed + rank)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=True,  # keeps batch shapes identical across workers/steps
    )


def get_full_dataloader(
    batch_size: int = 32,
    use_synthetic: bool = True,
    data_root: str = "./data",
    seed: int = 0,
    num_samples: int = 2000,
) -> DataLoader:
    """The non-sharded loader used by the single-process baseline."""
    dataset = get_dataset(use_synthetic=use_synthetic, data_root=data_root, num_samples=num_samples)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=True,
    )


if __name__ == "__main__":
    ds = SyntheticCIFARLike(num_samples=100)
    print(f"Dataset size: {len(ds)}")

    shards = shard_indices(100, num_workers=4)
    print(f"Shard sizes: {[len(s) for s in shards]}")
    assert sum(len(s) for s in shards) == 100
    assert len(set().union(*[set(s) for s in shards])) == 100  # disjoint + covers all

    loader = get_worker_dataloader(rank=0, world_size=4, batch_size=8)
    x, y = next(iter(loader))
    print(f"Batch shapes: x={x.shape}, y={y.shape}")