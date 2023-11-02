import os
from hashlib import md5

from PIL import Image
from optimum.onnxruntime import ORTStableDiffusionXLPipeline
from ai_worker.util import gunzip, download_file, url_to_tempfile

class SDXL:
    def __init__(self, conf):
        self.conf = conf
        self.base = None
        self.model = None
        self.load("stabilityai/stable-diffusion-xl-base-1.0")
        # unload from gpu
        self.base = None
        self.model = None

    async def load(self, model):
        if model != self.model:
            tmp = url_to_tempfile(self.conf, model, prefix="sdxl.")
            
            if not os.path.exists(tmp):
                url = None
                if model == "stabilityai/stable-diffusion-xl-base-1.0":
                    url = "https://gputopia.s3.amazonaws.com/models/sdxl.tar.gz"
                if url:
                    await download_file(url, tmp + ".tar.gz")
                    gunzip(tmp + ".tar.gz")
                    self.base = ORTStableDiffusionXLPipeline.from_pretrained(tmp)
                else:
                    self.base = ORTStableDiffusionXLPipeline.from_pretrained(model,
                                                                             trust_remote_code=False,
                                                                             export=True)
                    self.base.save_pretrained(tmp)
            else:
                self.base = ORTStableDiffusionXLPipeline.from_pretrained(tmp)
            self.base.to("cuda")
            self.model = model

    def temp_file(self, name, wipe=False):
        ret = os.path.join(self.conf.tmp_dir, name)
        return ret

    async def handle_req(self, req):
        await self.load("stabilityai/stable-diffusion-xl-base-1.0")
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
