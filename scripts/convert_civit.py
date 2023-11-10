import subprocess
import sys
from diffusers import StableDiffusionXLPipeline
import hashlib

if __name__ == "__main__":
    modid = sys.argv[1]
    name = sys.argv[2]
    url = "gputopia/" + name
    hashname = "sdxl." + hashlib.md5(url.encode()).hexdigest()
    subprocess.run("wget -O tmp.st https://civitai.com/api/download/models/" + modid, shell=True)
    pipe = StableDiffusionXLPipeline.from_single_file("tmp.st")
    pipe.save_pretrained(hashname)
    subprocess.run("tar -cf - {hashname}/ | pigz -9 - > {hashname}.tar.gz", shell=True)
    subprocess.run("aws s3 cp {hashname}.tar.gz s3://gputopia/models/{name}.tar.gz", shell=True)
