"""End-to-end smoke test: the pipeline stands up.

Replace the stub with a real check once a model exists: load configs/inference.yaml,
run one fixture image through ml.inference, and assert output shape/type/range
sanity (not exact values).
"""
from scripts.utilities.config import load_config


def test_inference_config_loads():
    cfg = load_config("inference")
    assert cfg["img_size"] > 0
    assert 0.0 <= cfg["conf_threshold"] <= 1.0


def test_inference_pipeline_stub():
    # TODO: replace with real single-image inference once a model is exported.
    assert True
