"""Category coverage and preference-gap calculations for CLI commands."""

from __future__ import annotations

from typing import Any, Mapping, Optional, cast


def aggregate_categories(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group catalog entries by normalized category name."""
    buckets: dict[str, dict[str, Any]] = {}
    for entry in entries:
        raw = entry.get("category")
        display = (
            raw.strip() if isinstance(raw, str) and raw.strip() else "Uncategorized"
        )
        key = display.casefold()
        bucket = buckets.setdefault(key, {"category": display, "count": 0})
        if bucket["category"] == "Uncategorized" and display != "Uncategorized":
            bucket["category"] = display
        bucket["count"] += 1
    return buckets


def preference_category_stats(preference_service: Any) -> dict[str, dict[str, Any]]:
    """Return normalized category rating statistics from an optional service."""
    try:
        profile = preference_service.export_profile()
    except Exception:  # pragma: no cover - defensive against custom services
        return {}

    categories = getattr(profile, "categories", {})
    if not isinstance(categories, Mapping):
        return {}

    stats: dict[str, dict[str, Any]] = {}
    for name, raw_bucket in categories.items():
        if not isinstance(name, str):
            continue
        bucket = _object(cast(object, raw_bucket))
        if not bucket:
            continue
        key = name.strip().casefold() or "uncategorized"
        stats[key] = {
            "avg_rating": _safe_average(bucket),
            "negative_ratio": _safe_negative_ratio(bucket),
        }
    return stats


def build_gap_records(
    categories: Mapping[str, Mapping[str, Any]],
    threshold: int,
    preference_data: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build renderer-ready coverage records from category statistics."""
    records: list[dict[str, Any]] = []
    for key, bucket in categories.items():
        count = bucket.get("count", 0)
        pref_stats = preference_data.get(key)
        avg_rating = pref_stats.get("avg_rating") if pref_stats else None
        negative_ratio = pref_stats.get("negative_ratio") if pref_stats else None
        flags: list[str] = []
        if threshold and count < threshold:
            flags.append("⚠ Below target")
        if negative_ratio is not None and negative_ratio >= 0.5:
            flags.append("⚠ Negative trend")
        records.append(
            {
                "category": bucket.get("category", key),
                "count": count,
                "avg_rating": avg_rating,
                "negative_ratio": negative_ratio,
                "status": "OK" if not flags else " / ".join(dict.fromkeys(flags)),
            }
        )
    return sorted(records, key=lambda record: (record["count"], record["category"]))


def _object(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return cast(dict[str, Any], value)
    return {}


def _safe_average(bucket: Mapping[str, Any]) -> Optional[float]:
    rating_count = _float(bucket.get("rating_count"))
    rating_sum = _float(bucket.get("rating_sum"))
    if rating_count:
        return (rating_sum or 0.0) / rating_count
    return None


def _safe_negative_ratio(bucket: Mapping[str, Any]) -> Optional[float]:
    count = _float(bucket.get("count"))
    negatives = _float(bucket.get("negatives"))
    if count:
        return (negatives or 0.0) / count
    return None


def _float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None
