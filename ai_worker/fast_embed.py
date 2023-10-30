import json

from fastembed.embedding import FlagEmbedding as Embedding

DEFAULT_MAX_LENGTH = 512

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"


class FastEmbed:
    def __init__(self, conf):
        self.conf = conf
        self.embedding_model = Embedding(model_name=DEFAULT_MODEL, max_length=DEFAULT_MAX_LENGTH)

    def embed(self, req: dict):
        model = req["model"]
        max_length = req.get("max_length", 512)

        if self.embedding_model.model_name != model or self.embedding_model._max_length != max_length: # noqa
            # swap out model
            self.embedding_model = Embedding(model_name=model, max_length=max_length)

        docs = req["input"]
        if isinstance(docs, str):
            docs = [docs]

        # todo: better toks count
        toks = int(len(json.dumps(docs))/2.5)

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
