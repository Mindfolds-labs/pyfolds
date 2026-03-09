from pathlib import Path

from pyfolds.serving.tf_serving import (
    build_inference_payload,
    build_tf_serving_command,
    payload_to_json,
    prepare_versioned_saved_model,
)


def test_prepare_serving_layout(tmp_path: Path):
    src = tmp_path / "saved_model"
    (src / "variables").mkdir(parents=True)
    (src / "saved_model.pb").write_text("x", encoding="utf-8")

    dst = prepare_versioned_saved_model(src, tmp_path / "serving", model_name="pyfolds", version=3)
    assert dst.exists()
    assert (dst / "saved_model.pb").exists()


def test_command_and_payload(tmp_path: Path):
    cmd = build_tf_serving_command(serving_root=tmp_path, model_name="pyfolds")
    assert "tensorflow/serving" in cmd

    payload = build_inference_payload([[1.0, 2.0]])
    assert "instances" in payload
    assert payload_to_json(payload).startswith("{")
