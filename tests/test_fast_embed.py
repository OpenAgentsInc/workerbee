import pytest
import json
from fastembed.embedding import FlagEmbedding as Embedding
from ai_worker.fast_embed import FastEmbed, DEFAULT_MODEL  # Assuming FastEmbed is in your_module.py
from unittest.mock import patch, MagicMock

from ai_worker.main import Config


class TestFastEmbed:

    @pytest.fixture
    def fastembed_instance(self):
        conf = Config()
        return FastEmbed(conf)

    def test_embed_single_string(self, fastembed_instance):
        req = {"input": "hello world", "model": DEFAULT_MODEL}
        result = fastembed_instance.embed(req)
        expected_tokens = int(len(json.dumps([req["input"]])) / 2.5)

        assert result["object"] == "list"
        assert result["usage"]["prompt_tokens"] == expected_tokens
        assert result["usage"]["total_tokens"] == expected_tokens
        assert result["data"][0]["object"] == "embedding"
        assert result["data"][0]["embedding"]
        assert result["data"][0]["index"] == 0
