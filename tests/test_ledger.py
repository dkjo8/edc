from edc.ledger import RunRecord, append, config_hash, read_all


def test_append_and_read_roundtrip(tmp_path):
    path = tmp_path / "ledger.jsonl"
    rec = RunRecord(
        run_id="abc123",
        timestamp="2026-07-24T00:00:00+00:00",
        resolved_config={"run": {"seed": 0}, "model": {"latent_dim": 16}},
        task="arithmetic",
        split="id",
        seed=0,
        metrics={"acc": 0.5},
    )
    row = append(rec, path=path)
    assert row["run_id"] == "abc123"
    assert row["config_hash"] == config_hash(rec.resolved_config)

    rows = read_all(path)
    assert len(rows) == 1
    assert rows[0]["metrics"]["acc"] == 0.5
    assert "env" in rows[0] and "python" in rows[0]["env"]


def test_config_hash_is_order_independent():
    a = {"x": 1, "y": {"b": 2, "a": 3}}
    b = {"y": {"a": 3, "b": 2}, "x": 1}
    assert config_hash(a) == config_hash(b)
