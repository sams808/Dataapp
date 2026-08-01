"""
qt_worker.py — minimal background-worker helper (QThreadPool + QRunnable)
for the operations heavy enough to freeze the UI thread: Multi-Fit batch
runs and HT-XRD series peak tracking.

Contract for callers: the submitted function must be PURE computation —
read every widget value BEFORE submitting, touch no Qt widgets inside the
worker. Results come back on the main thread via the on_done callback
(signal-delivered), which is where all UI updates belong.

set_synchronous(True) makes run_in_thread execute inline — used by the
test suite so tests stay deterministic without wait-loops; production code
never calls it.
"""
from __future__ import annotations

import traceback
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

_SYNCHRONOUS = False


def set_synchronous(value: bool) -> None:
    global _SYNCHRONOUS
    _SYNCHRONOUS = bool(value)


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)


class Worker(QRunnable):
    def __init__(self, fn: Callable, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            self.signals.error.emit(traceback.format_exc())
            return
        self.signals.finished.emit(result)


def run_in_thread(
    fn: Callable, on_done: Callable[[Any], None],
    on_error: Optional[Callable[[str], None]] = None,
    *args, **kwargs,
) -> Worker:
    """Run fn(*args, **kwargs) off the UI thread; deliver its return value
    to on_done (or the formatted traceback to on_error) on the main thread.
    Keep a reference to the returned Worker only if you need to inspect it —
    the thread pool owns its lifetime."""
    worker = Worker(fn, *args, **kwargs)
    worker.signals.finished.connect(on_done)
    if on_error is not None:
        worker.signals.error.connect(on_error)
    if _SYNCHRONOUS:
        worker.run()
    else:
        QThreadPool.globalInstance().start(worker)
    return worker
