import os
import typing
import base64
import time
import asyncio
import torch
from transformers import StableDiffusionPipeline
from io import BytesIO
from hashlib import md5
from typing import Optional
import logging as log

from ai_worker.util import gunzip, download_file, url_to_tempfile

if typing.TYPE_CHECKING:
    from PIL import Image

class _SDXL:
    def __init__(self, pipe, conf):
        self.pipe = pipe
        self.conf = conf
        self.base = None
        self.model = None
       
    async def preload(self):
        # load then unload
        await self.load("stabilityai/stable-diffusion-xl-base-1.0", download_only=True)

    def unload(self):
        log.info("unloading sdxl")
        self.base = None
        self.model = None

    async def load(self, model, download_only=False):
        loop = asyncio.get_running_loop()

        if model != self.model:
            tmp = url_to_tempfile(self.conf, model, prefix="sdxl.")
            base = None
            if not os.path.exists(tmp):
                url = None
                if model == "stabilityai/stable-diffusion-xl-base-1.0":
                    url = "https://gputopia.s3.amazonaws.com/models/sdxl.tar.gz"
                if url:
                    await download_file(url, tmp + ".tar.gz")
                    await loop.run_in_executor(None, lambda: gunzip(tmp + ".tar.gz"))
                    if not download_only:
                        base = self.pipe.from_pretrained(tmp)
                else:
                    base = await loop.run_in_executor(None, lambda: self.pipe.from_pretrained(model, trust_remote_code=False))
                    base.save_pretrained(tmp)
            else:
                if not download_only:
                    base = self.pipe.from_pretrained(tmp)

            if not download_only:
                log.info("moving sdxl to cuda")
                base.to("cuda")
                self.base = base
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
        
        images = self.run(req.get("prompt"), n, w, h, req.get("hyperparameters", {}))
       
        data = []
        
        for idx, img in enumerate(images):
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue())
            data.append({"object": "image", "index": idx, "data": img_str})
        
        ret = {
               "created": int(time.time()),
               "object": "list",
               "data": data
               }
        return ret

    def run(
            self,
            prompt: str,
            n: int,
            width: int,
            height: int,
            hp: dict
    ) -> list["Image"]:
        ret = self.base(prompt=prompt, width=width, height=height, num_images_per_prompt=n, num_inference_steps=hp.get("steps", 50))
        return ret.images

def SDXL(conf) -> Optional[_SDXL]:
    try:
        if "sdxl" not in conf.enable:
            return None
        pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", use_auth_token=True)
        if not torch.cuda.is_available() and not os.environ.get("CI"):
            log.exception("sdxl not enabled, PyTorch does not see the GPU")
            return None
        return _SDXL(pipeline, conf)
    except ImportError:
        if os.environ.get("GPUTOPIA_DEBUG_IMPORT"):
            log.exception("sdxl not enabled")
        return None
