"""Timestamped stdout for HTCondor / cluster logs.

Pipeline role
-------------
Used by Steps 1–5 so progress lines are flushed and time-stamped. Not a
pipeline step.
"""

from datetime import datetime


def tprint(*args, **kwargs):
    """Print with a timestamp prefix; flush by default so logs update live."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    kwargs.setdefault("flush", True)
    print(f"[{ts}]", *args, **kwargs)
