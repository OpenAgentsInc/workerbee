import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import json
from ai_worker.fast_embed import FastEmbed, DEFAULT_MODEL

from ai_worker.main import Config


@pytest.fixture
def fastembed_onnx():
    conf = Config()
    yield FastEmbed(conf)


@pytest.fixture
def fastembed_mocked(monkeypatch):
    mod = MagicMock()
    monkeypatch.setitem(sys.modules, 'fastembed.embedding', mod)
    conf = Config()
    ret = FastEmbed(conf)
    ret.embedding_model.embed.return_value = [
        np.array([1, 2, 3])
    ]
    yield ret


@pytest.fixture(params=["fastembed_onnx", "fastembed_mocked"])
def fastembed_instance(request):
    if request.param == "fastembed_onnx" and not request.config.getoption("--run-onnx"):
        pytest.skip("fastembed_onnx requires --run-onnx option to run")
    return request.getfixturevalue(request.param)


def test_embed_single_string(fastembed_instance):
    req = {"input": "hello world", "model": DEFAULT_MODEL}
    result = fastembed_instance.embed(req)
    expected_tokens = int(len(json.dumps([req["input"]])) / 2.5)

    assert result["object"] == "list"
    assert result["usage"]["prompt_tokens"] == expected_tokens
    assert result["usage"]["total_tokens"] == expected_tokens
    assert result["data"][0]["object"] == "embedding"
    assert result["data"][0]["embedding"]
    assert result["data"][0]["index"] == 0
