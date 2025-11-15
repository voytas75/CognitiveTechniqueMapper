"""Preference aggregation and personalization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..db.preference_repository import PreferenceRepository


PreferenceBucket = Dict[str, float | int]


@dataclass
class PreferenceProfile:
    """Aggregated preference signals grouped by technique and category."""

    techniques: Dict[str, PreferenceBucket] = field(default_factory=dict)
    categories: Dict[str, PreferenceBucket] = field(default_factory=dict)
    totals: PreferenceBucket = field(
        default_factory=lambda: {
            "count": 0,
            "positives": 0,
            "negatives": 0,
            "neutral": 0,
            "rating_sum": 0.0,
            "rating_count": 0,
        }
    )


class PreferenceService:
    """Computes preference-aware adjustments for technique selection."""

    def __init__(self, repository: PreferenceRepository) -> None:
        self._repository = repository
        self._profile: PreferenceProfile | None = None

    def record_preference(
        self,
        *,
        technique: Optional[str],
        category: Optional[str],
        rating: Optional[int],
        sentiment: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Persist and cache a new preference signal."""

        resolved_sentiment = sentiment or self._derive_sentiment(rating)
        timestamp = self._current_timestamp()
        self._repository.insert(
            technique=technique.strip() if technique else None,
            category=category.strip() if category else None,
            rating=rating,
            sentiment=resolved_sentiment,
            notes=notes.strip() if notes else None,
            created_at=timestamp,
        )
        self._profile = None

    def score_adjustment(self, metadata: dict) -> float:
        """Return a score delta to apply for the provided technique metadata."""

        profile = self._ensure_profile()
        technique = self._normalize(metadata.get("name"))
        category = self._normalize(metadata.get("category"))

        adjustment = 0.0
        if technique and technique in profile.techniques:
            adjustment += self._bucket_score(profile.techniques[technique], weight=0.15)
        if category and category in profile.categories:
            adjustment += self._bucket_score(profile.categories[category], weight=0.1)
        return adjustment

    def preference_summary(self) -> str:
        """Return a user-readable summary of current preferences."""

        profile = self._ensure_profile()
        if profile.totals["count"] == 0:
            return ""

        lines: list[str] = []
        top_tech = self._top_bucket(profile.techniques)
        if top_tech:
            name, bucket = top_tech
            avg = self._bucket_average(bucket)
            lines.append(
                f"Positive response to {name} (avg rating {avg:.1f})."
            )

        top_category = self._top_bucket(profile.categories)
        if top_category:
            name, bucket = top_category
            avg = self._bucket_average(bucket)
            lines.append(
                f"Favors {name} category techniques (avg rating {avg:.1f})."
            )

        negative = self._bottom_bucket(profile.techniques)
        if negative:
            name, bucket = negative
            lines.append(
                f"Watch for {name}; feedback trends negative."
            )
        return " ".join(lines)

    def export_profile(self) -> PreferenceProfile:
        """Expose the full preference profile for advanced consumers."""

        return self._ensure_profile()

    def preference_impacts(
        self, *, limit: int = 5
    ) -> dict[str, List[dict[str, float | int | None | str]]]:
        """Return score adjustments derived from stored preferences.

        Args:
            limit (int): Maximum number of entries to include per dimension.

        Returns:
            dict[str, list[dict[str, float | int | None | str]]]: Technique and category impact summaries.
        """

        profile = self._ensure_profile()
        resolved_limit = limit if limit > 0 else 5
        techniques = self._summarize_impacts(
            profile.techniques, weight=0.15, limit=resolved_limit
        )
        categories = self._summarize_impacts(
            profile.categories, weight=0.1, limit=resolved_limit
        )
        if not techniques and not categories:
            return {"techniques": [], "categories": []}
        return {"techniques": techniques, "categories": categories}

    def clear(self) -> None:
        """Remove all stored preferences and reset cached aggregates."""

        self._repository.delete_all()
        self._profile = None

    def _ensure_profile(self) -> PreferenceProfile:
        if self._profile is None:
            self._profile = self._compute_profile()
        return self._profile

    def _compute_profile(self) -> PreferenceProfile:
        profile = PreferenceProfile()
        rows = self._repository.fetch_all()
        for row in rows:
            technique = self._normalize(row.get("technique"))
            category = self._normalize(row.get("category"))
            rating = row.get("rating")
            sentiment = row.get("sentiment") or self._derive_sentiment(rating)

            self._accumulate(profile.totals, sentiment, rating)

            if technique:
                bucket = profile.techniques.setdefault(
                    technique, self._empty_bucket()
                )
                self._accumulate(bucket, sentiment, rating)
            if category:
                bucket = profile.categories.setdefault(
                    category, self._empty_bucket()
                )
                self._accumulate(bucket, sentiment, rating)
        return profile

    @staticmethod
    def _derive_sentiment(rating: Optional[int]) -> str:
        if rating is None:
            return "neutral"
        if rating >= 4:
            return "positive"
        if rating <= 2:
            return "negative"
        return "neutral"

    @staticmethod
    def _current_timestamp() -> str:
        from datetime import datetime, timezone

        return (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )

    @staticmethod
    def _normalize(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @staticmethod
    def _empty_bucket() -> PreferenceBucket:
        return {
            "count": 0,
            "positives": 0,
            "negatives": 0,
            "neutral": 0,
            "rating_sum": 0.0,
            "rating_count": 0,
        }

    @staticmethod
    def _accumulate(bucket: PreferenceBucket, sentiment: str, rating: Optional[int]) -> None:
        bucket["count"] = int(bucket.get("count", 0)) + 1
        sentiment_key = f"{sentiment}s" if sentiment in {"positive", "negative"} else "neutral"
        if sentiment_key not in bucket:
            # fall back to neutral if schema mismatch
            sentiment_key = "neutral"
        bucket[sentiment_key] = int(bucket.get(sentiment_key, 0)) + 1
        if rating is not None:
            bucket["rating_sum"] = float(bucket.get("rating_sum", 0.0)) + float(rating)
            bucket["rating_count"] = int(bucket.get("rating_count", 0)) + 1

    @staticmethod
    def _bucket_average(bucket: PreferenceBucket) -> float:
        count = int(bucket.get("rating_count", 0))
        if count == 0:
            return 0.0
        return float(bucket.get("rating_sum", 0.0)) / count

    @staticmethod
    def _bucket_score(bucket: PreferenceBucket, *, weight: float) -> float:
        count = int(bucket.get("count", 0))
        if count == 0:
            return 0.0
        positives = int(bucket.get("positives", 0))
        negatives = int(bucket.get("negatives", 0))
        net = positives - negatives
        normalized = net / count
        rating_avg = PreferenceService._bucket_average(bucket)
        rating_offset = 0.0
        if int(bucket.get("rating_count", 0)) > 0:
            rating_offset = (rating_avg - 3.0) / 2.0
        combined = (normalized + rating_offset) / 2.0
        combined = max(min(combined, 1.0), -1.0)
        return weight * combined

    @staticmethod
    def _top_bucket(buckets: Dict[str, PreferenceBucket]) -> tuple[str, PreferenceBucket] | None:
        best: tuple[str, PreferenceBucket] | None = None
        best_score = float("-inf")
        for name, bucket in buckets.items():
            score = PreferenceService._bucket_score(bucket, weight=1.0)
            if score > best_score:
                best = (name, bucket)
                best_score = score
        if best_score <= 0:
            return None
        return best

    @staticmethod
    def _bottom_bucket(buckets: Dict[str, PreferenceBucket]) -> tuple[str, PreferenceBucket] | None:
        worst: tuple[str, PreferenceBucket] | None = None
        worst_score = float("inf")
        for name, bucket in buckets.items():
            score = PreferenceService._bucket_score(bucket, weight=1.0)
            if score < worst_score:
                worst = (name, bucket)
                worst_score = score
        if worst_score >= 0:
            return None
        return worst

    @staticmethod
    def _summarize_impacts(
        buckets: Dict[str, PreferenceBucket],
        *,
        weight: float,
        limit: int,
    ) -> List[dict[str, float | int | None | str]]:
        summaries: List[dict[str, float | int | None | str]] = []
        for name, bucket in buckets.items():
            adjustment = PreferenceService._bucket_score(bucket, weight=weight)
            raw_score = PreferenceService._bucket_score(bucket, weight=1.0)
            average = PreferenceService._bucket_average(bucket)
            rating_count = int(bucket.get("rating_count", 0))
            if adjustment == 0 and raw_score == 0:
                continue
            summaries.append(
                {
                    "name": name,
                    "adjustment": round(adjustment, 4),
                    "raw_score": round(raw_score, 4),
                    "count": int(bucket.get("count", 0)),
                    "positives": int(bucket.get("positives", 0)),
                    "negatives": int(bucket.get("negatives", 0)),
                    "average_rating": round(average, 2) if rating_count > 0 else None,
                }
            )

        summaries.sort(
            key=lambda entry: (abs(float(entry["adjustment"])), float(entry["adjustment"])),
            reverse=True,
        )
        if limit and len(summaries) > limit:
            return summaries[:limit]
        return summaries
