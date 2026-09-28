from anima_backend import (
    MODEL_FILE,
    TEXT_ENCODER_FILE,
    VAE_FILE,
    build_workflow,
)
import threading
import time
from tempfile import TemporaryDirectory
from unittest.mock import patch

from anima_backend import AnimaRuntime


def test_build_workflow_uses_anima_components():
    workflow = build_workflow(
        prompt="masterpiece, best quality, safe, 1girl",
        negative_prompt="low quality",
        width=812,
        height=1216,
        steps=30,
        cfg=4.0,
        sampler="euler",
        scheduler="sgm_uniform",
        seed=42,
    )

    assert workflow["1"]["inputs"]["unet_name"] == MODEL_FILE
    assert workflow["2"]["inputs"]["clip_name"] == TEXT_ENCODER_FILE
    assert workflow["3"]["inputs"]["vae_name"] == VAE_FILE
    assert workflow["6"]["inputs"] == {"width": 812, "height": 1216, "batch_size": 1}
    assert workflow["7"]["inputs"]["seed"] == 42
    assert workflow["9"]["class_type"] == "SaveImage"


def test_workflow_graph_references_existing_nodes():
    workflow = build_workflow(
        prompt="test",
        negative_prompt="bad",
        width=1024,
        height=1024,
        steps=28,
        cfg=3.5,
        sampler="er_sde",
        scheduler="beta",
        seed=0,
    )
    node_ids = set(workflow)
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids


def test_wait_until_ready_allows_early_request():
    with TemporaryDirectory() as tmp_path, patch.dict(
        "os.environ", {"ANIMA_RUNTIME_ROOT": tmp_path}
    ):
        runtime = AnimaRuntime()

        def mark_ready():
            time.sleep(0.02)
            runtime._set_state("ready", "ok", ready=True)

        threading.Thread(target=mark_ready, daemon=True).start()
        runtime.wait_until_ready(timeout=3)
        assert runtime.state.ready
