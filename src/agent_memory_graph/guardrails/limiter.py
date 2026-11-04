"""Concurrency limiting with queue deadlines, and per-store circuit breakers.

Both primitives are event-loop local: every state change happens synchronously
between ``await`` points, so no lock is needed (and none is taken).
"""

import asyncio
import math
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from agent_memory_graph.domain.errors import (
    MemoryGraphError,
    StoreTimeoutError,
    StoreUnavailableError,
    VectorSearchTimeoutError,
)

__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "CircuitStats",
    "ConcurrencyLimiter",
    "LimiterStats",
    "is_store_failure",
]

_STORE_FAILURES: Final = (StoreUnavailableError, StoreTimeoutError, VectorSearchTimeoutError)


# ---------------------------------------------------------------------------
# Concurrency limiter
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LimiterStats:
    """Point-in-time view of a limiter."""

    capacity: int
    in_use: int
    waiting: int
    acquired_total: int
    rejected_total: int


class ConcurrencyLimiter:
    """Bounded semaphore whose waiters give up after ``queue_timeout_seconds``.

    A caller that cannot get a permit in time receives the error built by
    ``on_queue_timeout`` (e.g. ``StoreTimeoutError`` or ``ExtractionTimeoutError``)
    instead of queueing unboundedly behind a saturated dependency.
    """

    def __init__(
        self,
        name: str,
        *,
        max_concurrency: int,
        queue_timeout_seconds: float,
        on_queue_timeout: Callable[[float], MemoryGraphError],
    ) -> None:
        if max_concurrency < 1:
            msg = "max_concurrency must be at least 1"
            raise ValueError(msg)
        if not queue_timeout_seconds > 0:
            msg = "queue_timeout_seconds must be positive"
            raise ValueError(msg)
        self.name = name
        self._capacity = max_concurrency
        self._queue_timeout = queue_timeout_seconds
        self._on_queue_timeout = on_queue_timeout
        self._semaphore = asyncio.BoundedSemaphore(max_concurrency)
        self._in_use = 0
        self._waiting = 0
        self._acquired = 0
        self._rejected = 0

    def stats(self) -> LimiterStats:
        """Return the current occupancy and lifetime counters."""
        return LimiterStats(
            capacity=self._capacity,
            in_use=self._in_use,
            waiting=self._waiting,
            acquired_total=self._acquired,
            rejected_total=self._rejected,
        )

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        """Hold one permit for the duration of the ``async with`` block."""
        self._waiting += 1
        try:
            async with asyncio.timeout(self._queue_timeout):
                await self._semaphore.acquire()
        except TimeoutError:
            self._rejected += 1
            raise self._on_queue_timeout(self._queue_timeout) from None
        finally:
            self._waiting -= 1
        self._in_use += 1
        self._acquired += 1
        try:
            yield
        finally:
            self._in_use -= 1
            self._semaphore.release()


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------
class CircuitState(StrEnum):
    """``closed``: calls flow; ``open``: calls fail fast; ``half_open``: one probe."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(StoreUnavailableError):
    """Raised without calling the store while its circuit is open (503 + Retry-After)."""


@dataclass(frozen=True, slots=True)
class CircuitStats:
    """Point-in-time view of a breaker."""

    state: CircuitState
    consecutive_failures: int
    opened_total: int
    rejected_total: int


def is_store_failure(exc: BaseException) -> bool:
    """Whether ``exc`` means the store itself is unhealthy (unreachable or too slow).

    Rejections of the request (integrity errors, invalid input, ...) prove the store
    is answering and therefore do not count against it.
    """
    return isinstance(exc, _STORE_FAILURES) and not isinstance(exc, CircuitOpenError)


class CircuitBreaker:
    """Trips open after ``failure_threshold`` consecutive store failures.

    While open, calls raise :class:`CircuitOpenError` immediately. After
    ``reset_timeout_seconds`` the breaker turns half-open and lets exactly one probe
    through: success closes it, failure re-opens it for another full timeout.
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        reset_timeout_seconds: float = 30.0,
        is_failure: Callable[[BaseException], bool] = is_store_failure,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            msg = "failure_threshold must be at least 1"
            raise ValueError(msg)
        if not reset_timeout_seconds > 0:
            msg = "reset_timeout_seconds must be positive"
            raise ValueError(msg)
        self.name = name
        self._threshold = failure_threshold
        self._reset_timeout = reset_timeout_seconds
        self._is_failure = is_failure
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False
        self._opened_total = 0
        self._rejected = 0

    @property
    def state(self) -> CircuitState:
        """Current state; ``open`` lapses into ``half_open`` once the timeout passes."""
        if self._opened_at is None:
            return CircuitState.CLOSED
        if self._clock() - self._opened_at >= self._reset_timeout:
            return CircuitState.HALF_OPEN
        return CircuitState.OPEN

    def stats(self) -> CircuitStats:
        """Return the current state and lifetime counters."""
        return CircuitStats(
            state=self.state,
            consecutive_failures=self._failures,
            opened_total=self._opened_total,
            rejected_total=self._rejected,
        )

    def _reject(self) -> CircuitOpenError:
        self._rejected += 1
        elapsed = self._clock() - (self._opened_at or 0.0)
        retry_after = max(1, math.ceil(self._reset_timeout - elapsed))
        return CircuitOpenError(self.name, retry_after_seconds=retry_after)

    def _record_failure(self) -> None:
        self._failures += 1
        state = self.state
        if state is CircuitState.OPEN:
            return  # a straggler that started before the trip: already open
        if state is CircuitState.HALF_OPEN or self._failures >= self._threshold:
            self._opened_at = self._clock()
            self._opened_total += 1

    def _record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def check(self) -> None:
        """Fail fast with :class:`CircuitOpenError` if a call would be rejected now.

        Lets callers refuse work *before* queueing for a permit; :meth:`guard` repeats
        the check, so a breaker that opens while the caller queued still rejects it.
        """
        state = self.state
        if state is CircuitState.OPEN or (
            state is CircuitState.HALF_OPEN and self._probe_in_flight
        ):
            raise self._reject()

    @asynccontextmanager
    async def guard(self) -> AsyncIterator[None]:
        """Run the ``async with`` body under the breaker, or fail fast when open."""
        state = self.state
        if state is CircuitState.OPEN:
            raise self._reject()
        probing = state is CircuitState.HALF_OPEN
        if probing:
            if self._probe_in_flight:
                raise self._reject()
            self._probe_in_flight = True
        try:
            yield
        except asyncio.CancelledError:
            raise  # neither success nor failure: the caller gave up
        except BaseException as exc:
            if self._is_failure(exc):
                self._record_failure()
            else:
                self._record_success()
            raise
        else:
            self._record_success()
        finally:
            if probing:
                self._probe_in_flight = False
