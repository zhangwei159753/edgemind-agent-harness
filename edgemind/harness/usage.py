"""Small allowlisted helpers for provider token-usage metadata."""
from __future__ import annotations


_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def usage_from_response(response) -> dict[str, int] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    result = {}
    for field in _TOKEN_FIELDS:
        value = usage.get(field) if isinstance(usage, dict) else getattr(usage, field, None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[field] = value
    return result or None


def sum_usage(items) -> dict[str, int] | None:
    totals = {field: 0 for field in _TOKEN_FIELDS}
    found = False
    for item in items:
        if not isinstance(item, dict):
            continue
        found = True
        for field in _TOKEN_FIELDS:
            value = item.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                totals[field] += value
    return totals if found else None
