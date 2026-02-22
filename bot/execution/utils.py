"""Shared helpers for the execution sub-package."""


def strip_mode_prefix(identifier: str) -> str:
    """Remove a ``DRY-`` or ``PAPER-`` prefix from a deal / order identifier.

    >>> strip_mode_prefix("DRY-abc123")
    'abc123'
    >>> strip_mode_prefix("plain_id")
    'plain_id'
    """
    if "-" not in identifier:
        return identifier
    prefix, rest = identifier.split("-", 1)
    if prefix in {"DRY", "PAPER"} and rest:
        return rest
    return identifier
