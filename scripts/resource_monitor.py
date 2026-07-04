"""
Resource monitoring for a running benchmark subprocess.

Samples CPU%, RAM%, live process count, and network I/O for a process
and all of its children (the benchmark CLI process plus every worker
process it spawns) -- giving a live view of what's actually happening
on the machine while a benchmark runs, similar to what real distributed
training frameworks expose during a job.
"""

import psutil


class ProcessTreeMonitor:

    def __init__(self):
        self._processes: dict = {}  # pid -> psutil.Process, reused across calls

    def sample(self, root_pid: int) -> dict:
        try:
            root = self._processes.get(root_pid) or psutil.Process(root_pid)
            self._processes[root_pid] = root
        except psutil.NoSuchProcess:
            return {"cpu_percent": 0.0, "ram_percent": 0.0, "process_count": 0}

        try:
            children = root.children(recursive=True)
        except psutil.NoSuchProcess:
            children = []

        current_pids = {root_pid}
        procs_to_sample = [root]
        for child in children:
            current_pids.add(child.pid)
            # Reuse the cached object for this pid if we've seen it
            # before; otherwise cache this newly-discovered one.
            cached = self._processes.get(child.pid, child)
            self._processes[child.pid] = cached
            procs_to_sample.append(cached)

        cpu_total = 0.0
        alive_count = 0
        for p in procs_to_sample:
            try:
                cpu_total += p.cpu_percent(interval=None)
                alive_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        self._processes = {pid: p for pid, p in self._processes.items() if pid in current_pids}

        try:
            ram_percent = psutil.virtual_memory().percent
        except Exception:
            ram_percent = 0.0

        logical_cores = psutil.cpu_count(logical=True) or 1
        return {
            "cpu_percent": min(cpu_total, 100.0 * logical_cores),
            "ram_percent": ram_percent,
            "process_count": alive_count,
        }


def sample_network_io() -> dict:
    """
    System-wide network I/O counters (cumulative bytes since boot).
    RingSync's actual traffic is loopback-only (127.0.0.1 between
    workers), and loopback interface naming isn't reliably consistent
    across platforms (especially Windows), so this reports total system
    network I/O.
    """
    try:
        counters = psutil.net_io_counters()
        return {"bytes_sent": counters.bytes_sent, "bytes_recv": counters.bytes_recv}
    except Exception:
        return {"bytes_sent": 0, "bytes_recv": 0}


if __name__ == "__main__":
    import os
    import time

    print("Sampling this process's own tree for 5 seconds (first reading is expected to be near-0, by design)...")
    pid = os.getpid()
    monitor = ProcessTreeMonitor()
    for i in range(5):
        stats = monitor.sample(pid)
        net = sample_network_io()
        print(f"[{i}] CPU: {stats['cpu_percent']:.1f}%  RAM: {stats['ram_percent']:.1f}%  "
              f"Processes: {stats['process_count']}  Net sent: {net['bytes_sent']}")
        # busy-work so there's actually measurable CPU load to detect
        [x**2 for x in range(2_000_000)]
        time.sleep(0.5)
