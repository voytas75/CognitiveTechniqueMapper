from dataclasses import dataclass

from src.core.orchestrator import Orchestrator


@dataclass
class DummyWorkflow:
    name: str = "dummy"

    def run(self, context: dict) -> dict:
        return {"echo": context}


def test_orchestrator_executes_registered_workflow():
    workflow = DummyWorkflow()
    orchestrator = Orchestrator(workflows={workflow.name: workflow})

    result = orchestrator.execute("dummy", {"value": 42})
    assert result["echo"]["value"] == 42
