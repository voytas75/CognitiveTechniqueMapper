from pathlib import Path

from src.db.preference_repository import PreferenceRepository
from src.db.sqlite_client import SQLiteClient
from src.services.preference_service import PreferenceService


def test_preference_service_records_and_scores(tmp_path: Path) -> None:
    db_path = tmp_path / "prefs.db"
    client = SQLiteClient(db_path)
    client.initialize_schema()

    repository = PreferenceRepository(sqlite_client=client)
    service = PreferenceService(repository=repository)

    service.record_preference(
        technique="Decisional Balance",
        category="Decision Making",
        rating=5,
        notes="Excellent fit.",
    )
    service.record_preference(
        technique="Six Thinking Hats",
        category="Creative Thinking",
        rating=2,
        notes="Too cumbersome.",
    )

    profile = service.export_profile()
    assert profile.techniques["Decisional Balance"]["positives"] == 1
    assert profile.techniques["Six Thinking Hats"]["negatives"] == 1

    positive_adjustment = service.score_adjustment(
        {"name": "Decisional Balance", "category": "Decision Making"}
    )
    negative_adjustment = service.score_adjustment(
        {"name": "Six Thinking Hats", "category": "Creative Thinking"}
    )

    assert positive_adjustment > 0
    assert negative_adjustment < 0
    summary = service.preference_summary()
    assert "Decisional Balance" in summary

    service.clear()
    cleared_profile = service.export_profile()
    assert cleared_profile.totals["count"] == 0
