"""
Orchestration server.

Responsibilities, deliberately scoped:
  1. Registry: workers register their (rank, host, port) at job start.
  2. Heartbeat monitoring: workers ping periodically; a background
     thread checks for ranks that have gone silent past a timeout.
  3. Eviction: when a rank times out, it's removed from the active
     roster and its data range gets folded back into the survivors
     (via fault_handler.redistribute_shards).
  4. Roster serving: workers query GET_ROSTER at epoch boundaries to
     learn the current active worker list, so they rebuild their ring
     topology against reality rather than a stale, job-start snapshot.

What this deliberately does NOT do: recover a gradient contribution
lost mid-all-reduce, or resume an evicted worker without a fresh
registration. Detection and epoch-boundary redistribution is the
scoped claim; anything finer-grained is out of scope and is called out
as such in the README.
"""

import argparse
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

from ringsync.network.protocol import send_json, recv_json, MsgType
from ringsync.network.socket_utils import set_common_socket_options, ConnectionClosedError
from ringsync.orchestrator.fault_handler import redistribute_shards, build_rank_remap


@dataclass
class WorkerInfo:
    rank: int
    host: str
    port: int
    shard_range: Tuple[int, int]
    last_heartbeat: float = field(default_factory=time.time)
    alive: bool = True


class OrchestratorServer:
    def __init__(self, control_port: int, heartbeat_timeout_s: float = 8.0, check_interval_s: float = 2.0):
        self.control_port = control_port
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.check_interval_s = check_interval_s

        self.workers: Dict[int, WorkerInfo] = {}
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._events_log = []  # human-readable log of eviction events, exposed for tests/metrics

    def register_worker(self, rank: int, host: str, port: int, shard_range: Tuple[int, int]) -> None:
        with self.lock:
            self.workers[rank] = WorkerInfo(rank=rank, host=host, port=port, shard_range=shard_range)
        self._log(f"registered rank {rank} at {host}:{port}, shard={shard_range}")

    def heartbeat(self, rank: int) -> bool:
        """Returns False if the rank is unknown or already evicted."""
        with self.lock:
            info = self.workers.get(rank)
            if info is None or not info.alive:
                return False
            info.last_heartbeat = time.time()
            return True

    def get_active_roster(self) -> Dict[int, WorkerInfo]:
        with self.lock:
            return {r: w for r, w in self.workers.items() if w.alive}

    def _log(self, msg: str) -> None:
        entry = f"[orchestrator] {msg}"
        print(entry)
        self._events_log.append(entry)

    def check_timeouts_once(self) -> Set[int]:
        """
        Single pass over the registry looking for stale heartbeats.
        Returns the set of ranks newly evicted in this pass. Exposed as
        its own method (rather than only running inside the background
        loop) so tests can call it deterministically instead of racing
        a background thread against sleep() calls.
        """
        now = time.time()
        newly_dead = set()
        with self.lock:
            for rank, info in self.workers.items():
                if info.alive and (now - info.last_heartbeat) > self.heartbeat_timeout_s:
                    info.alive = False
                    newly_dead.add(rank)

        if newly_dead:
            self._handle_evictions(newly_dead)
        return newly_dead

    def _handle_evictions(self, dead_ranks: Set[int]) -> None:
        with self.lock:
            shard_ranges = {r: w.shard_range for r, w in self.workers.items() if w.alive or r in dead_ranks}
            new_ranges = redistribute_shards(shard_ranges, dead_ranks)
            remap = build_rank_remap(shard_ranges, dead_ranks)

            # Apply new shard ranges + renumber survivors
            old_infos = {r: w for r, w in self.workers.items()}
            self.workers = {}
            for old_rank, new_rank in remap.items():
                info = old_infos[old_rank]
                info.rank = new_rank
                info.shard_range = new_ranges[new_rank]
                self.workers[new_rank] = info

        for r in dead_ranks:
            self._log(f"EVICTED rank {r} (heartbeat timeout > {self.heartbeat_timeout_s}s). "
                       f"Shard redistributed among {len(remap)} survivors.")

    def _timeout_monitor_loop(self) -> None:
        while not self._stop.is_set():
            self.check_timeouts_once()
            self._stop.wait(self.check_interval_s)

    def _handle_connection(self, conn: socket.socket) -> None:
        set_common_socket_options(conn)
        try:
            while True:
                msg_type, payload = recv_json(conn)

                if msg_type == MsgType.REGISTER:
                    self.register_worker(
                        rank=payload["rank"], host=payload["host"], port=payload["port"],
                        shard_range=tuple(payload["shard_range"]),
                    )
                    send_json(conn, MsgType.REGISTER_ACK, {"ok": True})

                elif msg_type == MsgType.HEARTBEAT:
                    ok = self.heartbeat(payload["rank"])
                    send_json(conn, MsgType.HEARTBEAT_ACK, {"ok": ok})

                elif msg_type == MsgType.JSON and payload.get("request") == "GET_ROSTER":
                    roster = self.get_active_roster()
                    send_json(conn, MsgType.JSON, {
                        "world_size": len(roster),
                        "addresses": {str(r): [w.host, w.port] for r, w in roster.items()},
                        "shard_ranges": {str(r): list(w.shard_range) for r, w in roster.items()},
                    })

                elif msg_type == MsgType.SHUTDOWN:
                    send_json(conn, MsgType.SHUTDOWN, {"ok": True})
                    break

        except ConnectionClosedError:
            pass
        finally:
            conn.close()

    def serve_forever(self) -> None:
        listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_sock.bind(("0.0.0.0", self.control_port))
        listen_sock.listen(16)
        self._log(f"listening on port {self.control_port}")

        monitor_thread = threading.Thread(target=self._timeout_monitor_loop, daemon=True)
        monitor_thread.start()

        try:
            while not self._stop.is_set():
                listen_sock.settimeout(1.0)
                try:
                    conn, _addr = listen_sock.accept()
                except socket.timeout:
                    continue
                threading.Thread(target=self._handle_connection, args=(conn,), daemon=True).start()
        finally:
            listen_sock.close()

    def stop(self) -> None:
        self._stop.set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RingSync orchestrator server")
    parser.add_argument("--port", type=int, default=16000)
    parser.add_argument("--heartbeat-timeout", type=float, default=8.0)
    args = parser.parse_args()

    server = OrchestratorServer(control_port=args.port, heartbeat_timeout_s=args.heartbeat_timeout)
    server.serve_forever()
