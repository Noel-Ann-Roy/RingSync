"""
Fault handling logic: what happens to a dead worker's data shard.

Kept as pure functions with no sockets/threads involved, so the
redistribution logic itself can be unit-tested directly against
index math -- exactly like ring_allreduce.py's approach. The
orchestrator server (server.py) is a thin wrapper that calls into this
module when it detects a timeout.

Scope, stated plainly: this handles the case described in the project
spec -- a worker goes silent (heartbeat timeout), and its remaining
data for the epoch gets redistributed among the survivors, with the
ring reconfigured to the smaller world size for subsequent epochs.
This is NOT the harder unsolved research problem of recovering a
worker's *in-flight* gradient contribution mid-step, or resuming a
dead worker later without restarting it -- that's genuine open
research territory (stale-gradient handling, elastic training), and
this project doesn't claim to solve it.
"""

from typing import Dict, List, Set, Tuple


def redistribute_shards(
    original_shard_ranges: Dict[int, Tuple[int, int]],
    dead_ranks: Set[int],
) -> Dict[int, Tuple[int, int]]:
    """
    Given each rank's original [start, end) index range into the
    dataset, and a set of ranks now considered dead, return new
    contiguous [start, end) ranges for the surviving ranks that
    together cover the exact same overall index range -- the dead
    workers' data doesn't disappear, it gets folded back in and
    re-split evenly among whoever is left.

    Surviving ranks are renumbered contiguously starting at 0, since
    after eviction the ring needs dense rank numbering (0..new_world_size-1)
    to form a valid ring topology -- gaps in rank numbering would break
    the (rank+1) % world_size neighbor math.
    """
    if not original_shard_ranges:
        return {}

    all_starts = [r[0] for r in original_shard_ranges.values()]
    all_ends = [r[1] for r in original_shard_ranges.values()]
    total_start, total_end = min(all_starts), max(all_ends)
    total_len = total_end - total_start

    survivors = sorted(r for r in original_shard_ranges if r not in dead_ranks)
    new_world_size = len(survivors)

    if new_world_size == 0:
        return {}

    base = total_len // new_world_size
    remainder = total_len % new_world_size

    new_ranges = {}
    cursor = total_start
    for new_rank, old_rank in enumerate(survivors):
        size = base + (1 if new_rank < remainder else 0)
        new_ranges[new_rank] = (cursor, cursor + size)
        cursor += size

    return new_ranges


def build_rank_remap(
    original_shard_ranges: Dict[int, Tuple[int, int]],
    dead_ranks: Set[int],
) -> Dict[int, int]:
    """
    Maps old_rank -> new_rank for survivors (dead ranks are absent from
    the result). Workers need this to know their new identity in the
    reconfigured ring.
    """
    survivors = sorted(r for r in original_shard_ranges if r not in dead_ranks)
    return {old_rank: new_rank for new_rank, old_rank in enumerate(survivors)}


if __name__ == "__main__":
    # rank 0: [0,25), rank 1: [25,50), rank 2: [50,75), rank 3: [75,100)
    original = {0: (0, 25), 1: (25, 50), 2: (50, 75), 3: (75, 100)}

    new_ranges = redistribute_shards(original, dead_ranks={1})
    print(f"After rank 1 dies: {new_ranges}")

    # conservation check: total indices covered must be unchanged
    total_before = sum(e - s for s, e in original.values())
    total_after = sum(e - s for s, e in new_ranges.values())
    assert total_before == total_after, "lost or duplicated data indices!"

    # contiguity + full coverage check
    sorted_ranges = sorted(new_ranges.values())
    assert sorted_ranges[0][0] == 0
    assert sorted_ranges[-1][1] == 100
    for i in range(len(sorted_ranges) - 1):
        assert sorted_ranges[i][1] == sorted_ranges[i + 1][0]

    remap = build_rank_remap(original, dead_ranks={1})
    print(f"Rank remap after rank 1 dies: {remap}")
    assert remap == {0: 0, 2: 1, 3: 2}

    print("fault_handler sanity checks OK")
