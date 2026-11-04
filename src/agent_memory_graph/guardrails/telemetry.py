"""In-process telemetry: counters, latency distributions and live guardrail gauges.

Lock-free by construction: the service runs on one event loop and every update is a
synchronous read-modify-write with no ``await`` in between, so updates cannot
interleave. (Do not share one :class:`Telemetry` across threads or event loops.)
Memory is bounded: each latency series keeps fixed buckets plus a ring buffer of the
most recent samples for percentiles.
"""

import math
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Final

from agent_memory_graph.guardrails.limiter import CircuitBreaker, ConcurrencyLimiter

__all__ = ["COUNTERS", "LATENCY_BUCKETS_SECONDS", "LatencyStats", "Telemetry"]

LATENCY_BUCKETS_SECONDS: Final = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
_RESERVOIR: Final = 1024

COUNTERS: Final = (
    "ingest_requests",
    "ingest_failures",
    "query_requests",
    "query_failures",
    "entities_indexed",
    "relations_indexed",
    "candidates_pruned",
    "suspicious_inputs",
    "query_fallbacks",
    "fallback_hits",
    "empty_contexts",
    "traversal_hops_pruned",
)
"""Counters always present in a snapshot (others appear once incremented)."""


class LatencyStats:
    """Count, sum, extremes, cumulative buckets and recent-sample percentiles."""

    def __init__(self, buckets: tuple[float, ...] = LATENCY_BUCKETS_SECONDS) -> None:
        self._bounds = buckets
        self._bucket_counts = [0] * len(buckets)
        self._recent: deque[float] = deque(maxlen=_RESERVOIR)
        self.count = 0
        self.total = 0.0
        self.minimum = math.inf
        self.maximum = 0.0

    def observe(self, seconds: float) -> None:
        """Record one duration."""
        self.count += 1
        self.total += seconds
        self.minimum = min(self.minimum, seconds)
        self.maximum = max(self.maximum, seconds)
        self._recent.append(seconds)
        for index, bound in enumerate(self._bounds):
            if seconds <= bound:
                self._bucket_counts[index] += 1

    def percentile(self, fraction: float) -> float | None:
        """Nearest-rank percentile over the recent samples (``None`` when empty)."""
        if not self._recent:
            return None
        ordered = sorted(self._recent)
        rank = max(1, math.ceil(fraction * len(ordered)))
        return ordered[rank - 1]

    def snapshot(self) -> dict[str, Any]:
        """JSON-ready summary (seconds)."""
        return {
            "count": self.count,
            "sum_seconds": self.total,
            "min_seconds": self.minimum if self.count else None,
            "max_seconds": self.maximum if self.count else None,
            "mean_seconds": self.total / self.count if self.count else None,
            "p50_seconds": self.percentile(0.50),
            "p95_seconds": self.percentile(0.95),
            "p99_seconds": self.percentile(0.99),
            "buckets": {
                f"le_{bound:g}": count
                for bound, count in zip(self._bounds, self._bucket_counts, strict=True)
            }
            | {"le_inf": self.count},
        }


class Telemetry:
    """Collects service metrics and exposes them as one JSON-ready snapshot."""

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._started = clock()
        self._counters: dict[str, int] = dict.fromkeys(COUNTERS, 0)
        self._latency: dict[str, LatencyStats] = {}
        self._limiters: dict[str, ConcurrencyLimiter] = {}
        self._breakers: dict[str, CircuitBreaker] = {}

    def increment(self, name: str, amount: int = 1) -> None:
        """Add ``amount`` to counter ``name``."""
        self._counters[name] = self._counters.get(name, 0) + amount

    def counter(self, name: str) -> int:
        """Return counter ``name`` (0 if never incremented)."""
        return self._counters.get(name, 0)

    def latency(self, name: str) -> LatencyStats:
        """Return the latency series ``name``, creating it on first use."""
        return self._latency.setdefault(name, LatencyStats())

    @contextmanager
    def track(self, operation: str) -> Iterator[None]:
        """Count ``<operation>_requests`` / ``_failures`` and time the block."""
        self.increment(f"{operation}_requests")
        started = self._clock()
        try:
            yield
        except BaseException:
            self.increment(f"{operation}_failures")
            raise
        finally:
            self.latency(operation).observe(self._clock() - started)

    def register_limiter(self, limiter: ConcurrencyLimiter) -> None:
        """Report ``limiter`` occupancy in snapshots."""
        self._limiters[limiter.name] = limiter

    def register_breaker(self, breaker: CircuitBreaker) -> None:
        """Report ``breaker`` state in snapshots."""
        self._breakers[breaker.name] = breaker

    def snapshot(self) -> dict[str, Any]:
        """Everything collected so far, as plain JSON types."""
        limiters = {name: limiter.stats() for name, limiter in sorted(self._limiters.items())}
        breakers = {name: breaker.stats() for name, breaker in sorted(self._breakers.items())}
        return {
            "uptime_seconds": self._clock() - self._started,
            "counters": dict(sorted(self._counters.items())),
            "latency": {name: s.snapshot() for name, s in sorted(self._latency.items())},
            "limiters": {
                name: {
                    "capacity": stats.capacity,
                    "in_use": stats.in_use,
                    "available": stats.capacity - stats.in_use,
                    "waiting": stats.waiting,
                    "acquired_total": stats.acquired_total,
                    "rejected_total": stats.rejected_total,
                }
                for name, stats in limiters.items()
            },
            "circuits": {
                name: {
                    "state": stats.state.value,
                    "consecutive_failures": stats.consecutive_failures,
                    "opened_total": stats.opened_total,
                    "rejected_total": stats.rejected_total,
                }
                for name, stats in breakers.items()
            },
        }
