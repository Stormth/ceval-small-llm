from pathlib import Path

from ceval_eval.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_all_experiment_configs_load_and_have_unique_ids():
    paths = sorted((ROOT / "configs").glob("experiments/*.yaml"))
    paths += sorted((ROOT / "configs").glob("ablations/*.yaml"))
    assert paths
    ids = []
    for path in paths:
        config = load_config(path)
        assert config["experiment"]["id"]
        assert config["method"]["name"] in {"single", "self_check", "self_consistency"}
        ids.append(config["experiment"]["id"])
    assert len(ids) == len(set(ids))
