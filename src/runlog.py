"""Write everything a run prints to a log file, in addition to the terminal.

    with RunLog(directory, label="r_simple_rectangle_seed1", header=[...]) as log:
        ...  # everything printed in here goes to the terminal AND the log

Both stdout and stderr are mirrored, so a player's own `print()`s are captured
along with the simulator's output and any error messages. The terminal output
is unchanged. The log also gets a header (when/how the run was started) and a
footer (how long it took), and the traceback if the run crashed.

Only Python-level output is mirrored; text a native library writes straight to
the terminal's file descriptor is not.
"""

import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Self


class _Tee:
    """Looks like the original stream, but also writes to the log file."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file

    def write(self, text):
        written = self._stream.write(text)
        self._log.write(text)
        return written

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):  # isatty, encoding, fileno, ...
        return getattr(self._stream, name)


def _safe(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label)


def _open_unique(directory: Path, stem: str):
    """Create `<stem>.log`, or `<stem>-1.log`, ... if that name is taken."""
    for n in range(1000):
        path = directory / (f"{stem}.log" if n == 0 else f"{stem}-{n}.log")
        try:
            # line-buffered: each line reaches the file as it is written, so
            # the log survives a run that is killed
            return path, open(path, "x", encoding="utf-8", buffering=1)
        except FileExistsError:
            continue
    raise FileExistsError(f"could not find a free log name for {stem!r} in {directory}")


class RunLog:
    def __init__(self, directory: Path | str, label: str, header: list[str]):
        self._directory = Path(directory)
        self._label = _safe(label)
        self._header = header
        self.path: Path | None = None
        self._file = None
        self._streams = None
        self._wall_start = 0.0
        self._cpu_start = 0.0

    def __enter__(self) -> Self:
        self._directory.mkdir(parents=True, exist_ok=True)
        started = datetime.now().astimezone()  # local time, timezone-aware
        stem = f"{started:%Y%m%d-%H%M%S}_{self._label}"
        self.path, self._file = _open_unique(self._directory, stem)
        self._file.write("# playpen run log\n")
        self._file.write(f"# started: {started:%Y-%m-%d %H:%M:%S} (local time)\n")
        for line in self._header:
            self._file.write(f"# {line}\n")
        self._file.write("# ----\n")

        self._streams = (sys.stdout, sys.stderr)
        sys.stdout = _Tee(sys.stdout, self._file)
        sys.stderr = _Tee(sys.stderr, self._file)
        self._wall_start = time.perf_counter()
        self._cpu_start = time.process_time()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = self._streams

        if exc_type is None:
            outcome = "finished normally"
        elif issubclass(exc_type, SystemExit):
            code = getattr(exc, "code", 0)
            outcome = "finished normally" if not code else f"exited with status {code}"
        elif issubclass(exc_type, KeyboardInterrupt):
            outcome = "interrupted"
        else:
            outcome = f"crashed: {exc_type.__name__}: {exc}"
            # the traceback goes to the log directly: Python itself prints it
            # to the terminal once this exception propagates, and printing it
            # here too would show it twice
            self._file.write("".join(traceback.format_exception(exc_type, exc, tb)))

        self._file.write("# ----\n")
        self._file.write(
            f"# {outcome}; wall time {time.perf_counter() - self._wall_start:.2f}s, "
            f"process CPU time {time.process_time() - self._cpu_start:.2f}s\n"
        )
        self._file.close()
        return False
