# technically we should load lots of models and gpu configurations to be sure we make good guesses
# probably someone can do a better job... and not guess, because llama 2 is perfectly predictable
# each model type will be different tho

from ai_worker.main import WorkerMain, Config

#@patch("ai_worker.main.pyopencl")

async def test_layer_est():
    config = Config()
    wm = WorkerMain(config)
    path = await wm.download_model("TheBloke/CodeLlama-7B-Instruct-GGUF:Q4_K_M")
    assert await wm.guess_layers(path) > 20



