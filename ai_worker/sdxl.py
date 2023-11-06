import os
import asyncio
import base64
import time
from io import BytesIO
import logging as log
from typing import Optional

from util import gunzip, download_file, url_to_tempfile
from diffusers import DiffusionPipeline, StableDiffusionPipeline
import torch

class SDXL:
    def __init__(self, conf, pipeline=DiffusionPipeline):
        self.conf = conf
        self.base = None
        self.model = None
        self.pipe = pipeline
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def preload(self):
        # Asynchronous preload of the model
        await self.load("stabilityai/stable-diffusion-xl-base-1.0", download_only=True)

    def unload(self):
        # Unload the model from memory
        log.info("Unloading SDXL")
        self.base = None
        self.model = None

    async def load(self, model, download_only=False):
        # Asynchronously load the model into memory
        loop = asyncio.get_running_loop()
        if model != self.model:
            tmp = url_to_tempfile(self.conf, model, prefix="sdxl_")
            if not os.path.exists(tmp):
                url = None
                if model == "stabilityai/stable-diffusion-xl-base-1.0":
                    url = "https://gputopia.s3.amazonaws.com/models/sdxl.tar.gz"
                if url:
                    await download_file(url, tmp + ".tar.gz")
                    await loop.run_in_executor(None, lambda: gunzip(tmp + ".tar.gz"))
                else:
                    base = await loop.run_in_executor(None, lambda: self.pipe.from_pretrained(model))
                    base.save_pretrained(tmp)
            if not download_only:
                log.info("Moving SDXL to CUDA")
                self.base = await loop.run_in_executor(None, lambda: self.pipe.from_pretrained(tmp))
                self.base.to(self.device)
                self.model = model

    def temp_file(self, name, wipe=False):
        # Handle temporary files
        ret = os.path.join(self.conf.tmp_dir, name)
        return ret

    async def handle_req(self, req):
        # Handle image generation requests
        await self.load("stabilityai/stable-diffusion-xl-base-1.0")
        
        sz = req.get("size", "1024x1024")
        w, h = map(int, sz.split("x"))
        n = req.get("n", 1)
        num_inference_steps = req.get("hyperparameters", {}).get("steps", 50)

        images = self.run(req["prompt"], n, w, h, num_inference_steps)

        data = []
        for idx, img in enumerate(images):
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            data.append({"object": "image", "index": idx, "data": img_str})

        return {
            "created": int(time.time()),
            "object": "list",
            "data": data
        }

    def run(self, prompt: str, n: int, width: int, height: int, num_inference_steps: int) -> list:
        # Generate and return images based on the prompt and parameters
        generated_images = self.base(prompt=prompt, width=width, height=height, num_images_per_prompt=n, num_inference_steps=num_inference_steps)
        return generated_images.images

def _SDXL(conf) -> Optional[SDXL]:
    try:
        if "sdxl" not in conf.enable:
            return None
        pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", use_auth_token=True)
        if not torch.cuda.is_available() and not os.environ.get("CI"):
            log.exception("SDXL not enabled, PyTorch does not see the GPU")
            return None
        return SDXL(conf, pipeline=pipeline)
    except ImportError as e:
        log.exception("SDXL not enabled: %s", e)
        return None