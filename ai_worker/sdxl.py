import os
import typing
import base64
import time
import asyncio

from io import BytesIO
from typing import Optional
import logging as log

from ai_worker.util import gunzip, download_file, url_to_tempfile

if typing.TYPE_CHECKING:
    from PIL import Image

class _SDXL:
    def __init__(self, *, pipe, torch, conf):
        self.pipe = pipe
        self.torch = torch
        self.conf = conf
        self.base = None
        self.model = None
        self.device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        self.loaded = False
        self.loading = False

    async def preload(self):
        # loasd then unload
        await self.load("stabilityai/stable-diffusion-xl-base-1.0", download_only=True)

    def unload(self):
        log.info("unloading sdxl")
        self.base = None
        self.model = None

    async def load(self, model, download_only=False):
        assert not self.loading
        self.loading = True
        loop = asyncio.get_running_loop()
        if model != self.model:
            tmp = url_to_tempfile(self.conf, model, prefix="sdxl.")
            base = None
            if not os.path.exists(tmp):
                url = None
               
                # todo: normalize this
                if model == "gputopia/sdxl":
                    model = "stabilityai/stable-diffusion-xl-base-1.0"
                
                if model == "stabilityai/stable-diffusion-xl-base-1.0":
                    url = "https://gputopia.s3.amazonaws.com/models/sdxl.tar.gz"
                
                if model.startswith("gputopia/"):
                    name = model[9:]
                    url = f"https://gputopia.s3.amazonaws.com/models/{name}.tar.gz"
                
                if url:
                    await download_file(url, tmp + ".tar.gz")
                    await loop.run_in_executor(None, lambda: gunzip(tmp + ".tar.gz"))
                    if not download_only:
                        base = self.pipe.from_pretrained(tmp)
                else:
                    base = await loop.run_in_executor(None, lambda: self.pipe.from_pretrained(model, trust_remote_code=False, export=True))
                    base.save_pretrained(tmp)
            else:
                if not download_only:
                    base = self.pipe.from_pretrained(tmp)

            if not download_only:
                log.info("moving sdxl to %s", self.device)
                base.to(self.device)
                self.base = base
                self.model = model
        self.loading = False
        self.loaded = True

    def temp_file(self, name, wipe=False):
        ret = os.path.join(self.conf.tmp_dir, name)
        return ret

    async def handle_req(self, req):
        await self.load(req["model"])
         
        sz = req.get("size", "1024x1024")
        w, h = sz.split("x")
        
        w = int(w)
        h = int(h)
        
        n = req.get("n", 1)
       
        prompt = req["prompt"]
        neg = req.get("negative_prompt")
        
        if not neg:
            neg = ["bad teeth", "cartoon", "blurry", "cropped", "ugly"]
            neg = [k for k in neg if k not in prompt]
            neg = ", ".join(neg)
        
        images = self.run(prompt, neg, n, w, h, req.get("hyperparameters", {}))
       
        data = []
        
        for idx, img in enumerate(images):
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            data.append({"b64_json": img_str})
        
        ret = {
               "created": int(time.time()),
               "data": data
              }
        return ret

    def run(
            self,
            prompt: str,
            negative_prompt: str,
            n: int,
            width: int,
            height: int,
            hp: dict
    ) -> list["Image"]:
        ret = self.base(prompt=prompt, negative_prompt=negative_prompt, width=width, height=height, num_images_per_prompt=n, num_inference_steps=hp.get("steps", 50))
        return ret.images


def SDXL(conf) -> Optional[_SDXL]:
    try:
        if "sdxl" not in conf.enable:
            return None

        from diffusers import StableDiffusionXLPipeline
        import torch
        assert torch.cuda.is_available() or os.environ.get("CI"), "Cuda not available, SDXL disabled"
        return _SDXL(pipe=StableDiffusionXLPipeline, torch=torch, conf=conf)
    except (ImportError, AssertionError):
        if os.environ.get("GPUTOPIA_DEBUG_IMPORT"):
            log.exception("sdxl not enabled")
        return None

