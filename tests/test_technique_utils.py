from src.services.technique_utils import compose_embedding_text


def test_compose_embedding_text_concatenates_fields() -> None:
    text = compose_embedding_text(
        {
            "description": "Describe technique",
            "core_principles": "Principles",
            "category": "Decision",
        }
    )
    assert "Describe technique" in text
    assert "Principles" in text
    assert "Decision" in text
