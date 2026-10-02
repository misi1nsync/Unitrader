import pandas as pd
import pytest

from unitrader.state import State


def test_write_then_read_roundtrip(tmp_path):
    state = State(tmp_path)
    df = pd.DataFrame({"symbol": ["BTCUSDT"], "close": [1.0]})
    path = state.write("latest_data.parquet", df)
    assert path == tmp_path / "latest_data.parquet"
    pd.testing.assert_frame_equal(state.read("latest_data.parquet"), df)
    assert [p.name for p in tmp_path.iterdir()] == ["latest_data.parquet"]


def test_failed_write_keeps_previous_file(tmp_path, monkeypatch):
    state = State(tmp_path)
    good = pd.DataFrame({"close": [1.0]})
    state.write("latest_data.parquet", good)

    def boom(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError):
        state.write("latest_data.parquet", pd.DataFrame({"close": [2.0]}))
    monkeypatch.undo()
    pd.testing.assert_frame_equal(state.read("latest_data.parquet"), good)
    assert [p.name for p in tmp_path.iterdir()] == ["latest_data.parquet"]


def test_rejects_paths_outside_state_dir(tmp_path):
    with pytest.raises(ValueError):
        State(tmp_path).path("../escape.parquet")


def test_json_roundtrip(tmp_path):
    state = State(tmp_path)
    state.write("pending_signal.json", {"status": "pending", "signals": [1, 2]})
    assert state.read("pending_signal.json") == {"status": "pending", "signals": [1, 2]}


def test_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError):
        State(tmp_path).write("notes.txt", {})
