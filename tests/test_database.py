import json
from qmi import db


def test_sqlite_experiment_and_prediction_roundtrip(tmp_path, monkeypatch):
    database_file = tmp_path / "test.sqlite3"
    monkeypatch.setenv("QMI_DATABASE_PATH", str(database_file))
    db.init_db()
    run_id = db.add_experiment("smoke run", "synthetic fixture", config={"seed": 42},
        metrics={"macro_dice": 0.8, "undefined_hd95": float("nan")})
    event_id = db.add_prediction_event("case-001", 4, 16, "test-model", {"dice": 0.7})
    runs = db.list_experiments()
    events = db.list_predictions()
    assert run_id == runs[0]["id"]
    assert runs[0]["config"]["seed"] == 42
    assert runs[0]["metrics"]["undefined_hd95"] is None
    assert event_id == events[0]["id"]
    assert events[0]["case_id"] == "case-001"
    assert database_file.exists()
