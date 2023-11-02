from io import BytesIO
from typing import Optional
import torch
from diffusers import DiffusionPipeline
from PIL import Image
from optimum.onnxruntime import ORTStableDiffusionXLPipeline

class SDXL:
    def __init__(self):
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Load the base model
        self.base = ORTStableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        # Save ONNX model in the same directory as the SDXL.py
        save_directory = "./sd_xl_base_onnx_model" 
        self.base.save_pretrained(save_directory)

        self._refiner = None

    @property
    def refiner(self):
        # This part may need to be updated if a compatible ONNX refiner model is available
        if self._refiner is None:
            # ... refiner loading code ...
            pass
        return self._refiner

    def run(
        self,
        prompt: str,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> Image:
        images = self._run(
            prompt=prompt,
            width=width,
            height=height
        )
        return images[0]

    def _run(
        self,
        prompt,
        width,
        height
    ):
        images = self.base(prompt=prompt, width=width, height=height).images
        return images
