import base64
import hashlib
import os
import tarfile
from httpx import AsyncClient
import llama_cpp

GGML_TYPE_MAP = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
}

GGML_INVERSE_MAP={v.lower():k for k, v in GGML_TYPE_MAP.items()}

USER_PREFIX = "user:"

def b64enc(byt):
    return base64.urlsafe_b64encode(byt).decode()


def b64dec(str_):
    return base64.urlsafe_b64decode(str_)


def quantize_gguf(fil, level):
    out = fil + "." + level
    fil = fil.encode("utf-8")
    out = out.encode("utf-8")
    level = GGML_INVERSE_MAP[level.lower()]
    qp = llama_cpp.llama_model_quantize_default_params()
    qp.ftype = level
    return_code = llama_cpp.llama_model_quantize(fil, out, qp)
    if return_code != 0:
        raise RuntimeError("Failed to quantize model")
    return out


def user_ft_name_to_url(name):
    if name.startswith(USER_PREFIX):
        sub = name[len(USER_PREFIX):]
    else:
        sub = name
    if not sub.endswith(".gguf"):
        sub = sub + ".gguf"
    name = f"https://gputopia-user-bucket.s3.amazonaws.com/{sub}"
    return name


def url_to_tempfile(conf, url, prefix=""):
    name = hashlib.md5(url.encode()).hexdigest()
    output_file = os.path.join(conf.tmp_dir, prefix + name)
    return output_file


def gzip(folder):
    """Tar gz the folder to 'folder.tar.gz'"""
    base_folder_name = os.path.basename(folder)
    with tarfile.open(f"{folder}.tar.gz", 'w:gz') as archive:
        archive.add(folder, arcname=base_folder_name)
    return f"{folder}.tar.gz"

def gunzip(archive_path):
    """Unzips 'folder.tar.gz' to 'folder', and removes the 'folder.tar.gz'"""

    with tarfile.open(archive_path, 'r:gz') as archive:
        archive.extractall(path=os.path.dirname(archive_path))
    os.remove(archive_path)
    folder_path = os.path.splitext(archive_path)[0]  # remove the .tar.gz extension
    return folder_path


async def download_file(url: str, output_file: str) -> str:
    if not os.path.exists(output_file):
        with open(output_file + ".tmp", "wb") as fh:
            async with AsyncClient() as cli:
                async with cli.stream("GET", url) as res:
                    res: Response
                    async for chunk in res.aiter_bytes():
                        fh.write(chunk)
        os.replace(output_file + ".tmp", output_file)
    return output_file

