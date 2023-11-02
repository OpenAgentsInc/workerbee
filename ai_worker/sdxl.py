import os
from hashlib import md5

from PIL import Image
from optimum.onnxruntime import ORTStableDiffusionXLPipeline


class SDXL:
    def __init__(self, conf):
        self.conf = conf
        self.base = None
        self.model = None
        self.load("stabilityai/stable-diffusion-xl-base-1.0")
        # unload model from ram at init time so the process starts ok
        self.base = None

    def load(self, model):
        if model != self.model:
            model_hash = md5(model.encode()).hexdigest()
            tmp = self.temp_file("sdxl." + model_hash)
            if not os.path.exists(tmp):
                self.base = ORTStableDiffusionXLPipeline.from_pretrained(model,
                                                                         trust_remote_code=False,
                                                                         export=True)
                self.base.save_pretrained(tmp)
            else:
                self.base = ORTStableDiffusionXLPipeline.from_pretrained(tmp)
            self.model = model

    def temp_file(self, name, wipe=False):
        ret = os.path.join(self.conf.tmp_dir, name)
        return ret

    def handle_req(self, req):
        self.load("stabilityai/stable-diffusion-xl-base-1.0")
        sz = req.get("size", "1024x1024")
        w, h = sz.split("x")
        w = int(w)
        h = int(h)
        n = req.get("n", 1)
        images = self.run(req.get("prompt"), n, w, h)
        ret = {"object": "list",
               "data": [{"object": "image", "index": idx, "data": img} for idx, img in enumerate(images)]
               }
        return ret

    def run(
            self,
            prompt: str,
            n: int,
            width: int,
            height: int
    ) -> list[Image]:
        images = self._run(
            prompt=prompt,
            width=width,
            height=height,
            n=n
        )
        return images

    def _run(
            self,
            *,
            prompt,
            width,
            height,
            n
    ):
        images = self.base(prompt=prompt, width=width, height=height, num_images_per_prompt=n).images
        return images
