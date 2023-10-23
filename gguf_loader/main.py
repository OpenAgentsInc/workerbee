import argparse
import os.path
import sys

from dotenv import load_dotenv
import os
import boto3

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub import HfFileSystem
from gguf_loader.convert_llama_ggml_to_gguf import main as convert_to_gguf_main
from gguf_loader.convert import main as pytorch_to_gguf_main
import logging as log


def pytorch_to_gguf(path):
    dest = path + "/ggml-model-f16.gguf"
    sys.argv = ["convert", path, "--outfile", dest + ".tmp"]
    pytorch_to_gguf_main()
    os.replace(dest + ".tmp", dest)
    return dest


def convert_to_gguf(file):
    dest = file + ".gguf"
    if os.path.exists(dest) and os.path.getsize(dest):
        return dest
    sys.argv = ["convert-to-gguf", "-i", file, "-o", dest + ".tmp", "--eps", "1e-5"]
    if "70b" in dest.lower():
        sys.argv += ["--gqa", "8"]
    convert_to_gguf_main()
    os.replace(dest + ".tmp", dest)
    return dest


def get_size(name):
    typ, hf, fil = pick_file(name)
    if not fil:
        assert False, "not sizaable: %s" % name
    return fil["size"]


def pick_file(name):
    parts = name.split(":", 1)
    if len(parts) == 1:
        hf, filt = parts[0], ""
    else:
        hf, filt = parts

    fs = HfFileSystem()
    lst = fs.ls(hf)
    if filt:
        # sometimes quantization binaries have extension like q_4, etc.
        # we don't want to download all of them!
        lst = [f for f in lst if filt in f["name"]]

    gguf = [f for f in lst if "gguf" in f["name"]]

    if not gguf:
        log.info("no %s gguf, searching for ggml", name)
        ggml = [f for f in lst if "ggml" in f["name"]]
        try_conv = [f for f in lst if f["name"].endswith("/config.json")]

        if len(ggml) > 1:
            ggml = [f for f in lst if f["name"].endswith(".bin")]

        if len(ggml) > 1:
            raise ValueError("Multiple files match, please specify a better filter")

        if len(ggml):
            return "ggml", hf, ggml[0]

        if len(try_conv):
            return "pytorch", hf, ""

        raise ValueError("Can't find gguf, ggml or config.json")

    if len(gguf) > 1:
        raise ValueError("Multiple files match, please specify a better filter")

    return "gguf", hf, gguf[0]


def download_gguf(name):
    typ, repo_id, fil = pick_file(name)

    # todo: manage this way better
    #       remove old
    #       don't rely on cache, use our own
    #       remove ggml/tmp/etc
    #       add conversions for pth

    if typ == "pytorch":
        log.debug("downloading...")
        # use hf so we get a free cache
        path = snapshot_download(repo_id=repo_id, resume_download=True)
        return pytorch_to_gguf(path)

    if typ == "ggml":
        base = os.path.basename(fil["name"])
        log.debug("downloading...")
        # use hf so we get a free cache
        path = hf_hub_download(repo_id=repo_id, filename=base, resume_download=True)
        return convert_to_gguf(path)

    base = os.path.basename(fil["name"])
    log.debug("downloading...")
    return hf_hub_download(repo_id=repo_id, filename=base)


# Load environment variables from .env file
load_dotenv()


# Get AWS credentials from environment variables


def upload_to_s3(path, dest):
    # Split 'dest' into bucket and key (filename)
    bucket_name, key = dest.split("/", 1)

    aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
    aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
    aws_region = os.getenv('AWS_REGION')

    # Initialize a session using Amazon S3
    s3 = boto3.client('s3',
                      aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key,
                      region_name=aws_region)

    # Perform the upload
    s3.upload_file(path, bucket_name, key)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download a specific gguf model from hf, suitable for llama_cpp")
    parser.add_argument("--s3", help="upload to s3, must have .env with access key", action="store")
    parser.add_argument("--s3-region", help="s3 region, or can read from AWS_REGION", action="store")
    parser.add_argument("model", help="hugging face model name and optional [:filter]")
    args = parser.parse_args(args=argv)

    path = download_gguf(args.model)

    if args.s3:
        s3 = args.s3
        if args.s3_region:
            os.environ["AWS_REGION"] = args.s3_region

        if s3.endswith("/"):
            # give it a nice name
            s3 += args.model.replace("/", ".").replace(":", ".") + ".ggml"

        # useful if you're caching these conversions and names centrally somewhere
        upload_to_s3(path, s3)

        print(s3)
    else:
        print(path)
