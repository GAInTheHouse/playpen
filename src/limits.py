"""A CPU-time limit for player code.

CPU time is the time the processor spends executing the process (all its
threads), not the time on the clock: a player that sleeps or waits uses almost
none, and one that keeps four cores busy for a minute uses four.

    with CpuLimit(300) as cpu:
        construction = player.build_enclosure()
    print(cpu.used)

If the player is still running when its CPU budget is spent, `PlayerTimeout`
is raised inside its code. It is a `BaseException`, like `KeyboardInterrupt`,
so a player's own `except Exception:` can't swallow it.

Limits of this approach, which is in-process and so cooperative:
- It can only interrupt when Python code is running. A single native call
  that itself runs past the limit (one enormous geometry operation) is
  interrupted when it returns.
- Only this process's CPU counts. Work done in other processes the player
  starts (multiprocessing, joblib) is not included.
- Where the OS can't deliver the timer signal (Windows), nothing interrupts the
  player; the limit is checked when it returns, so an overrun is still
  reported but a player that never returns would hang the run.
- It uses signals, so it must run on the main thread.
"""

import signal
import time
from typing import Self

# True where we can interrupt a running player (Unix: needs SIGPROF/setitimer)
CAN_INTERRUPT = hasattr(signal, "setitimer") and hasattr(signal, "SIGPROF")

# setitimer rejects absurdly large values; this is about 3 years
_MAX_SECONDS = 1e8

# If the player swallows the interrupt, it fires again this many CPU seconds later
_REPEAT_SECONDS = 1.0


class PlayerTimeout(BaseException):
    """The player used up its CPU budget."""

    def __init__(self, used: float, limit: float):
        super().__init__(f"{used:.1f}s of CPU time used, limit is {limit:g}s")
        self.used = used
        self.limit = limit


class CpuLimit:
    """Context manager: raises `PlayerTimeout` if the code inside uses more
    than `seconds` of CPU time. `used` holds the CPU time spent inside,
    once the block has ended (however it ended)."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.used = 0.0
        self._start = 0.0
        self._armed = False
        self._previous_handler = None

    def _elapsed(self) -> float:
        return time.process_time() - self._start

    def _on_timer(self, signum, frame):
        # a signal that arrives after the block ended must do nothing
        if self._armed:
            raise PlayerTimeout(self._elapsed(), self.seconds)

    def __enter__(self) -> Self:
        self._start = time.process_time()
        self._armed = True
        if CAN_INTERRUPT:
            self._previous_handler = signal.signal(signal.SIGPROF, self._on_timer)
            signal.setitimer(
                signal.ITIMER_PROF, min(self.seconds, _MAX_SECONDS), _REPEAT_SECONDS
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._armed = False  # first, so a late signal can't raise from here on
        if CAN_INTERRUPT:
            signal.setitimer(signal.ITIMER_PROF, 0)
            signal.signal(signal.SIGPROF, self._previous_handler)
        self.used = self._elapsed()
        if exc is None and self.used > self.seconds:
            # not interrupted, but over the limit all the same: the player
            # swallowed the interrupt, or nothing could interrupt it
            raise PlayerTimeout(self.used, self.seconds)
        return False
