from src.db.sqlite_client import SQLiteClient


def test_sqlite_client_initializes_schema(tmp_path) -> None:
    db_path = tmp_path / "techniques.db"
    client = SQLiteClient(db_path)
    client.initialize_schema()

    client.insert_technique(
        name="Test Technique",
        description="A test technique description.",
        origin_year=2024,
        creator="Tester",
        category="Test",
        core_principles="Testing",
    )

    rows = client.fetch_all()
    assert len(rows) == 1
    assert rows[0]["name"] == "Test Technique"
