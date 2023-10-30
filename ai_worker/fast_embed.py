import json
import os
import logging as log
from typing import Optional

DEFAULT_MAX_LENGTH = 512
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
MODEL_PREFIX = "fastembed:"


class _FastEmbed:
    def __init__(self, cls, conf):
        self.conf = conf
        self.embedding_class = cls
        self.embedding_model = cls(model_name=DEFAULT_MODEL, max_length=DEFAULT_MAX_LENGTH)

    def embed(self, req: dict):
        model = req["model"]

        if model.startswith(MODEL_PREFIX + ":"):
            model = model[len(MODEL_PREFIX) + 1:]

        max_length = req.get("max_length", 512)

        if self.embedding_model.model_name != model or self.embedding_model._max_length != max_length:  # noqa
            # swap out model
            self.embedding_model = self.embedding_class(model_name=model, max_length=max_length)

        docs = req["input"]
        if isinstance(docs, str):
            docs = [docs]

        # todo: better toks count
        toks = int(len(json.dumps(docs)) / 2.5)

        res = {
            "object": "list",
            "model": model,
            "usage": {
                "prompt_tokens": toks,
                "total_tokens": toks
            }
        }

        embed = [
            dict(
                object="embedding",
                embedding=nda.tolist(),
                index=i
            )
            for i, nda in enumerate(self.embedding_model.embed(docs))
        ]

        res["data"] = embed

        return res


def FastEmbed(*a) -> Optional[_FastEmbed]:
    try:
        from fastembed.embedding import FlagEmbedding as Embedding
        return _FastEmbed(Embedding, *a)
    except ImportError:
        if os.environ.get("GPUTOPIA_DEBUG_IMPORT"):
            log.exception("fine tuning not enabled")
        return None
