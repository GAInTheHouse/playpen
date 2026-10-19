"""Read the simulator's CPU budget without changing the shared player API.

On Unix the active CpuLimit timer is authoritative, including elapsed constructor
time and programmatic Game/GUI runs. Otherwise use the CLI limit/default. Direct
API callers can supply a limit explicitly. No timer or simulator state is changed.
"""

import math
import signal
import sys
import time
from dataclasses import dataclass

from src.constants import CPU_TIME_LIMIT_SECONDS
from src.limits import CpuLimit
from src.pieces import ConnectorType


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("CPU limit must be a finite positive number")
    return seconds


def active_framework_limit():
    """Return (configured limit, remaining CPU seconds) for our framework only."""
    if not hasattr(signal, "getitimer") or not hasattr(signal, "SIGPROF"):
        return None
    try:
        handler = signal.getsignal(signal.SIGPROF)
        owner = getattr(handler, "__self__", None)
        if not isinstance(owner, CpuLimit):
            return None
        remaining, _ = signal.getitimer(signal.ITIMER_PROF)
        if remaining > 0 and math.isfinite(owner.seconds):
            return owner.seconds, remaining
    except (OSError, ValueError):
        return None
    return None


def command_line_limit(argv):
    """Read just --cpu-limit (including argparse-style abbreviations), not other flags."""
    found = None
    for index, argument in enumerate(argv):
        flag, separator, value = argument.partition("=")
        if len(flag) <= 2 or not "--cpu-limit".startswith(flag):
            continue
        try:
            found = positive_seconds(value if separator else argv[index + 1])
        except (ValueError, IndexError):
            # main.py already validates CLI input. Ignore unrelated/malformed
            # arguments when Player6 is used directly by a test or another app.
            continue
    return found


@dataclass(frozen=True)
class RunBudget:
    cpu_limit: float
    available_seconds: float
    safety_seconds: float
    usable_seconds: float
    deadline: float
    grid_deadline: float
    orthogonal_deadline: float
    grid_seconds: float
    beam_seconds: float
    local_seconds: float
    source: str


def make_budget(inventory, *, cpu_limit=None, stage_limits=None, argv=None):
    active = active_framework_limit()
    cli = command_line_limit(sys.argv[1:] if argv is None else argv)
    if cpu_limit is not None:
        limit, source = positive_seconds(cpu_limit), "explicit API"
    elif active is not None:
        limit, source = active[0], "simulator timer"
    elif cli is not None:
        limit, source = cli, "command line"
    else:
        limit, source = float(CPU_TIME_LIMIT_SECONDS), "simulator default"
    available = min(limit, active[1]) if active is not None else limit
    # Reserve 3%, at least 50ms for ordinary budgets, capped at 5s. Tiny or
    # almost-expired budgets reserve no more than 20% of what remains.
    safety = min(5.0, max(0.05, limit * 0.03), available * 0.2)
    usable = max(0.0, available - safety)
    has_beam = inventory.connectors.get(ConnectorType.DIAGONAL, 0) >= 2
    has_local = inventory.connectors.get(ConnectorType.DIAGONAL, 0) >= 4
    defaults = {
        "grid": usable * (0.30 if has_beam else 0.60),
        "beam": usable * 0.50 if has_beam else 0.0,
        "local": usable * 0.05 if has_local else 0.0,
    }
    limits = stage_limits or {}
    for name, value in limits.items():
        if name not in defaults:
            raise ValueError(f"Unknown stage budget: {name}")
        if value is not None:
            seconds = float(value)
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError(f"{name} budget must be finite and nonnegative")
            defaults[name] = seconds
    if not has_beam:
        defaults["beam"] = 0.0
    if not has_local:
        defaults["local"] = 0.0
    requested = sum(defaults.values())
    scale = min(1.0, usable / requested) if requested else 1.0
    grid, beam, local = (defaults[name] * scale for name in ("grid", "beam", "local"))
    now = time.process_time()
    deadline = now + usable
    return RunBudget(
        limit,
        available,
        safety,
        usable,
        deadline,
        now + grid,
        deadline - beam - local,
        grid,
        beam,
        local,
        source,
    )
