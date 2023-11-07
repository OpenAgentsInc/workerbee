# technically we should load lots of models and gpu configurations to be sure we make good guesses
# probably someone can do a better job... and not guess, because llama 2 is perfectly predictable
# each model type will be different tho

from unittest.mock import patch
from ai_worker.main import WorkerMain, Config
import ai_worker


async def test_layer_est():
    with patch.object(ai_worker.main.nvidia_smi, "getInstance") as gi:
        config = Config()
        wm = WorkerMain(config)
        gi().DeviceQuery.return_value = dict(
            count=1,
            driver_version="fake",
            gpu=[
                dict(
                    product_name="nvidia fake",
                    fb_memory_usage={"total": 4000},
                    clocks={"graphics_clock": 400, "unit": "ghz"},
                )
            ]
        )
        path = await wm.download_model("TheBloke/CodeLlama-7B-Instruct-GGUF:Q4_K_M")
        assert await wm.guess_layers(path) > 18


async def test_loader_scan_models():
    config = Config()
    wm = WorkerMain(config)
    await wm.download_model("TheBloke/CodeLlama-7B-Instruct-GGUF:Q4_K_M")
    assert "TheBloke/CodeLlama-7B-Instruct-GGUF:Q4_K_M" in wm.get_model_list()
    wm.llama_model = "foo"
    assert "foo" in wm.get_model_list()
    wm.llama_model = None
    before = len(wm.get_model_list())
    wm.llama_model = "TheBloke/CodeLlama-7B-Instruct-GGUF:Q4_K_M"
    assert len(wm.get_model_list()) == before
    assert wm.get_model_list()[0] == wm.llama_model

    srm = "https://some.random.model"
    wm.note_have(srm)
    assert srm in wm.get_model_info_from_config()

    # gone! because we verify that it really exists
    assert srm not in wm.get_model_list()