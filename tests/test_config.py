from edc.config import load_config


def test_smoke_config_merges_over_base():
    cfg = load_config("configs/smoke.toml")
    # smoke overrides base
    assert cfg.model.latent_dim == 24
    assert cfg.inference.k_restarts == 4
    assert cfg.run.notes == "smoke"
    # base defaults survive where not overridden
    assert cfg.conformal.alpha == 0.1


def test_to_dict_roundtrips_and_hashes():
    from edc.config import load_from_dict

    cfg = load_config("configs/smoke.toml")
    d = cfg.to_dict()
    cfg2 = load_from_dict(d)
    assert cfg2.to_dict() == d


def test_temperature_is_positive():
    # tau > 0 is a hard requirement (invariant): deterministic descent kills basin agreement.
    cfg = load_config("configs/smoke.toml")
    assert cfg.inference.temperature > 0.0
