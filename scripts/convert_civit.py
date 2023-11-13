import os
import subprocess
import sys
from diffusers import StableDiffusionXLPipeline
import hashlib

if __name__ == "__main__":
    modid = sys.argv[1]
    name = sys.argv[2]
    url = "gputopia/" + name
    hashname = "sdxl." + hashlib.md5(url.encode()).hexdigest()
    if not os.path.exists(f"{hashname}.st"):
        subprocess.run(f"wget -O {hashname}.st https://civitai.com/api/download/models/" + modid, shell=True)
    pipe = StableDiffusionXLPipeline.from_single_file(f"{hashname}.st", use_safetensors=True)
    pipe.save_pretrained(hashname)
    subprocess.run(f"tar -cf - {hashname}/ | pigz -9 - > {hashname}.tar.gz", shell=True)
    subprocess.run(f"aws s3 cp {hashname}.tar.gz s3://gputopia/models/{name}.tar.gz", shell=True)
