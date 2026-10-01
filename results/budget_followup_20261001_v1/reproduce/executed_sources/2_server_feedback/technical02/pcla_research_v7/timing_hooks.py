"""Call-scoped timing hooks; inactive outside an explicitly timed invocation."""
from contextlib import nullcontext
from contextvars import ContextVar
from functools import wraps

_ledger = ContextVar('pcla_cost_ledger', default=None)


def span(label):
    timing = _ledger.get()
    return timing.span(label) if timing is not None else nullcontext()


def timed(label):
    def decorate(function):
        @wraps(function)
        def invoke(*args, **kwargs):
            with span(label):
                return function(*args, **kwargs)
        return invoke
    return decorate


def measured(label, function, *args, **kwargs):
    with span(label):
        return function(*args, **kwargs)


def verification(function, *args, **kwargs):
    """Measure an audit-only repeat without charging its nested algorithm hooks."""
    timing = _ledger.get()
    if timing is None:
        return function(*args, **kwargs)
    token = _ledger.set(None)
    try:
        with timing.span('technical_equivalence_check'):
            return function(*args, **kwargs)
    finally:
        _ledger.reset(token)


def timing_context(label):
    """Bind a function's explicit timing keyword without changing its inputs."""
    def decorate(function):
        @wraps(function)
        def invoke(*args, **kwargs):
            timing = kwargs.get('timing')
            if timing is None:
                return function(*args, **kwargs)
            token = _ledger.set(timing)
            try:
                with timing.span(label):
                    return function(*args, **kwargs)
            finally:
                _ledger.reset(token)
        return invoke
    return decorate
