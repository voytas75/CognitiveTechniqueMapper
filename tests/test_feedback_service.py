from pathlib import Path

from src.core.feedback_manager import FeedbackManager
from src.services.feedback_service import FeedbackService
from src.db.feedback_repository import FeedbackRepository
from src.db.sqlite_client import SQLiteClient


class StubLLM:
    def __init__(self) -> None:
        self.captured_prompt: str | None = None
        self.workflow: str | None = None

    def invoke(self, workflow: str, prompt: str, **_: object) -> str:
        self.workflow = workflow
        self.captured_prompt = prompt
        return "Summary text"


def test_feedback_service_persists_and_summarizes(tmp_path: Path) -> None:
    db_path = tmp_path / "feedback.db"
    client = SQLiteClient(db_path)
    client.initialize_schema()

    repository = FeedbackRepository(sqlite_client=client)
    manager = FeedbackManager()
    llm = StubLLM()

    service = FeedbackService(
        feedback_manager=manager,
        llm_gateway=llm,  # type: ignore[arg-type]
        repository=repository,
        history_limit=10,
    )

    service.record_feedback("detect_technique", "Very helpful", rating=5)
    summary_payload = service.summarize_feedback()

    records = repository.fetch_recent()
    assert len(records) == 1
    assert records[0]["message"] == "Very helpful"

    assert summary_payload["workflow"] == "feedback_loop"
    assert summary_payload["summary"] == "Summary text"
    assert llm.workflow == "feedback_loop"
    assert llm.captured_prompt is not None and "Very helpful" in llm.captured_prompt
