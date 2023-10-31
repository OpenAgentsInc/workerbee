from io import BytesIO
from typing import Optional
import torch
from diffusers import DiffusionPipeline
from PIL import Image

class SDXL:
    def __init__(self):
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Load both base & refiner
        self.base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.half,
            variant="fp16",
            use_safetensors=True,
        )
        if cuda_available:
            self.base.to("cuda")
        else:
            self.device = torch.device("cpu")

        self._refiner = None

    @property
    def refiner(self):
        if self._refiner is None:
            pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                text_encoder_2=self.base.text_encoder_2,
                vae=self.base.vae,
                torch_dtype=torch.half,
                use_safetensors=True,
                variant="fp16",
            )
            if torch.cuda.is_available():
                pipe.to("cuda")
            self._refiner = pipe
        return self._refiner

    def run(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        guidance_scale: Optional[float] = 5.0,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = 50,
        high_noise_frac: Optional[float] = 0.8,
        use_refiner: Optional[bool] = True,
    ) -> Image:
        images = self._run(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            samples=1,
            seed=seed,
            num_inference_steps=num_inference_steps,
            high_noise_frac=high_noise_frac,
            use_refiner=use_refiner,
        )
        return images[0]

    def _run(
        self,
        prompt,
        negative_prompt,
        width,
        height,
        guidance_scale,
        samples,
        seed,
        num_inference_steps,
        high_noise_frac,
        use_refiner,
    ):
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        if samples > 1:
            prompt = [prompt] * samples
            if negative_prompt is not None:
                negative_prompt = [negative_prompt] * samples
            generator = [generator] * samples

        base_extra_kwargs = {}
        if use_refiner:
            base_extra_kwargs["output_type"] = "latent"
            base_extra_kwargs["denoising_end"] = high_noise_frac
        # Run both experts
        images = self.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            **base_extra_kwargs,
        ).images
        if use_refiner:
            images = self.refiner(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                denoising_start=high_noise_frac,
                image=images,
            ).images
        return images

