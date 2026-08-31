"""Timestamped print helper for cluster logs (HTCondor stdout)."""

from datetime import datetime


def tprint(*args, **kwargs):
    """Print with a timestamp prefix; flush by default so logs update live."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    kwargs.setdefault("flush", True)
    print(f"[{ts}]", *args, **kwargs)
