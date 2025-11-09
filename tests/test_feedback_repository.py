from datetime import datetime, timezone
from pathlib import Path

from src.db.feedback_repository import FeedbackRepository
from src.db.sqlite_client import SQLiteClient


def test_feedback_repository_persists_and_fetches(tmp_path: Path) -> None:
    db_path = tmp_path / "feedback.db"
    client = SQLiteClient(db_path)
    client.initialize_schema()

    repository = FeedbackRepository(sqlite_client=client)
    timestamp = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    repository.insert(
        workflow="detect_technique",
        message="Great recommendation",
        rating=5,
        created_at=timestamp,
    )

    records = repository.fetch_recent(limit=5)
    assert len(records) == 1
    record = records[0]
    assert record["workflow"] == "detect_technique"
    assert record["message"] == "Great recommendation"
    assert record["rating"] == 5
    assert record["created_at"]
