"""Small, CLI-scoped progress messages for otherwise quiet long-running stages."""
from contextlib import contextmanager
from contextvars import ContextVar
import sys
import threading
import time


_reporter = ContextVar('workflow_progress_reporter', default=None)


def format_count(value):
    """Format ordinary counts exactly and enormous counts with safe bounds.

    Progress must not convert an arbitrarily large integer into decimal text:
    Python limits that conversion and the output would be unreadable anyway.
    Integer-only lower/upper bounds on log10(2) give conservative powers of ten
    from the bit length, without converting the count to a float or a string.
    """
    if not isinstance(value, int) or value.bit_length() <= 1024:
        return f'{value:,}'
    if value < 0:
        return f'-({format_count(-value)})'
    bits = value.bit_length()
    denominator = 10**18
    lower = (bits - 1) * 301029995663981195 // denominator
    upper = (bits * 301029995663981196 + denominator - 1) // denominator
    return f'between 10^{lower:,} and 10^{upper:,}'


class _Reporter:
    def __init__(self, stream, interval):
        self.stream = stream
        self.interval = interval
        self.lock = threading.Lock()

    def write(self, message):
        with self.lock:
            print(message, file=self.stream, flush=True)


class _StageProgress:
    """In-memory counters only: workers never query a database or inspect logs."""

    def __init__(self, completed=0, total=None, unit='items'):
        self.completed, self.total, self.unit = completed, total, unit
        self.detail = None
        self.lock = threading.Lock()

    def update(self, completed, *, total=None, detail=None):
        with self.lock:
            self.completed = completed
            if total is not None:
                self.total = total
            self.detail = detail

    def description(self):
        with self.lock:
            if self.total is None:
                result = f'{format_count(self.completed)} {self.unit}' if self.completed else ''
            else:
                result = f'{format_count(self.completed)}/{format_count(self.total)} {self.unit}'
            if self.detail:
                result = f'{result}; {self.detail}' if result else self.detail
            return f' [{result}]' if result else ''


def message(text):
    """Write a progress summary when a caller explicitly enabled reporting."""
    reporter = _reporter.get()
    if reporter is not None:
        reporter.write(text)


@contextmanager
def report_progress(*, stream=None, interval=10.0):
    """Enable progress only inside this context; nested callers restore its owner."""
    if interval <= 0:
        raise ValueError('Progress interval must be positive')
    reporter = _Reporter(sys.stderr if stream is None else stream, interval)
    token = _reporter.set(reporter)
    try:
        yield
    finally:
        _reporter.reset(token)


@contextmanager
def stage(label, *, log=None, total=None, completed=0, unit='items'):
    """Report a stage and periodic elapsed time without reading or streaming logs."""
    reporter = _reporter.get()
    progress = _StageProgress(completed, total, unit)
    if reporter is None:
        yield progress
        return

    started = time.monotonic()
    suffix = f' (log: {log})' if log is not None else ''
    reporter.write(f'{label}...{suffix}{progress.description()}')
    stopped = threading.Event()

    def heartbeat():
        while not stopped.wait(reporter.interval):
            reporter.write(f'{label}: still running ({time.monotonic() - started:.1f}s elapsed){progress.description()}')

    worker = threading.Thread(target=heartbeat, name='madgraph-step-progress', daemon=True)
    status = 'done'
    try:
        worker.start()
        yield progress
    except (KeyboardInterrupt, EOFError):
        status = 'interrupted'
        raise
    except BaseException:
        status = 'failed'
        raise
    finally:
        stopped.set()
        if worker.ident is not None:
            worker.join()
        reporter.write(f'{label}: {status} ({time.monotonic() - started:.1f}s){progress.description()}')
