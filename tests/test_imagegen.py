import importlib
import sys
import logging as log
import pytest
import base64
from unittest.mock import MagicMock

from ai_worker.main import Config
from ai_worker.sdxl import SDXL


@pytest.fixture
def sdxl_cuda():
    conf = Config(enable=["sdxl"])
    yield SDXL(conf)


@pytest.fixture
def sdxl_mocked(monkeypatch):
    mod = MagicMock()
    monkeypatch.setitem(sys.modules, 'PIL', mod)
    monkeypatch.setitem(sys.modules, 'torch', mod)
    monkeypatch.setitem(sys.modules, 'diffusers', mod)

    import ai_worker.sdxl
    importlib.reload(ai_worker.sdxl)

    conf = Config(enable=["sdxl"])
    
    ret = SDXL(conf)
    ret.base = MagicMock()
    ret.base.return_value = MagicMock()
    ret.base.return_value.images = [MagicMock()]
    ret.model = "stabilityai/stable-diffusion-xl-base-1.0" 

    yield ret

    monkeypatch.undo()
    importlib.reload(ai_worker.sdxl)


@pytest.fixture(params=["cuda", "mocked"])
def sdxl_inst(request):
    if request.param == "cuda" and not request.config.getoption("--run-cuda"):
        pytest.skip("sdxl cuda requires --run-cuda option to run")
    return request.getfixturevalue("sdxl_" + request.param)


async def test_imagegen_simple(sdxl_inst: "SDXL"):
    req = {"model": "gputopia/nightvision-xl", "prompt": "a dog", "n": 1, "size": "1024x1024", "hyperparameters": {"steps": 4}}
    result = await sdxl_inst.handle_req(req)
    log.debug(result)
    assert result["created"]
    assert "b64_json" in result["data"][0]
    assert isinstance(result["data"][0]["b64_json"], str)
    if result["data"][0]["b64_json"]:
        assert base64.b64decode(result["data"][0]["b64_json"])
        
