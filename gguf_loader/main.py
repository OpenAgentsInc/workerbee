import argparse
import os.path
import re
import sys

from dotenv import load_dotenv
import os

from huggingface_hub import hf_hub_download, snapshot_download, scan_cache_dir, CacheNotFound
from huggingface_hub import HfFileSystem
from gguf_loader.convert_llama_ggml_to_gguf import main as convert_to_gguf_main
from gguf_loader.convert import main as pytorch_to_gguf_main
import logging as log


def pytorch_to_gguf(path):
    dest = pth_dest_path(path)
    sys.argv = ["convert", path, "--outfile", dest + ".tmp"]
    pytorch_to_gguf_main()
    os.replace(dest + ".tmp", dest)
    return dest


def pth_dest_path(path):
    dest = path + "/ggml-model-f16.gguf"
    return dest


def convert_to_gguf(file):
    dest = ggml_dest_path(file)
    if os.path.exists(dest) and os.path.getsize(dest):
        return dest
    sys.argv = ["convert-to-gguf", "-i", file, "-o", dest + ".tmp", "--eps", "1e-5"]
    if "70b" in dest.lower():
        sys.argv += ["--gqa", "8"]
    convert_to_gguf_main()
    os.replace(dest + ".tmp", dest)
    return dest


def ggml_dest_path(file):
    dest = file + ".gguf"
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


def get_model_abbr(repo, fp):
    m = re.match(r".*\d+b.*(q[_.-]?\d([_.-]k)?([_.-](m|s))?)([_.-]\d)?", fp, re.I)
    if m:
        filt = m[1]
        ret = repo + ":" + filt
    else:
        ret = repo
    return ret


def get_model_list():
    try:
        hf_cache_info = scan_cache_dir()
        ml = set()
        for ent in hf_cache_info.repos:
            repo = ent.repo_id
            rev = next(iter(ent.revisions))
            for f in rev.files:
                fp = str(f.file_path)
                if "ggml" in fp:
                    gg = ggml_dest_path(fp)
                    if os.path.exists(gg):
                        ml.add(get_model_abbr(repo, gg))
                if "gguf" in fp:
                    ml.add(get_model_abbr(repo, fp))
        return list(ml)
    except (FileNotFoundError, CacheNotFound):
        return []

# Load environment variables from .env file
load_dotenv()


# Get AWS credentials from environment variables


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download a specific gguf model from hf, suitable for llama_cpp")
    parser.add_argument("model", help="hugging face model name and optional [:filter]")
    args = parser.parse_args(args=argv)

    path = download_gguf(args.model)

    print(path)
