import argparse
import asyncio
import ctypes
import json
import logging
import logging.config
import multiprocessing
import os
import platform
import sys
import tempfile
import time
from hashlib import sha256, md5
from pprint import pprint
from typing import Optional, List, Literal, Any
from base64 import urlsafe_b64encode as b64encode, urlsafe_b64decode as b64decode
import psutil
import websockets
from httpx import Response, AsyncClient
from httpx_sse import aconnect_sse
from llama_cpp.server.app import Settings as LlamaSettings, create_app as create_llama_app
import llama_cpp.server.app
from whisper_cpp_python.server.app import Settings as WhisperSettings, create_app as create_whisper_app
import whisper_cpp_python.server.app
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pynvml.smi import nvidia_smi
import pyopencl
from dotenv import load_dotenv

from .util import user_ft_name_to_url, url_to_tempfile, USER_PREFIX, schedule_task

log = logging.getLogger(__name__)

try:
    from .fine_tune import FineTuner
except ImportError:
    if os.environ.get("GPUTOPIA_DEBUG_IMPORT"):
        log.exception("fine tuning not enabled")
    FineTuner = None

from ai_worker.sdxl import SDXL

from .fast_embed import FastEmbed, MODEL_PREFIX
from gguf_loader.main import get_size

from .gguf_reader import GGUFReader
from .key import PrivateKey
from .version import VERSION

APP_NAME = "gputopia"
ENV_PREFIX = APP_NAME.upper()

DEFAULT_COORDINATOR = "wss://queenbee.gputopia.ai/worker"

load_dotenv()


class Req(BaseModel):
    openai_url: str
    openai_req: dict


class GpuInfo(BaseModel):
    name: Optional[str]
    uuid: Optional[str] = None
    memory: Optional[float]
    clock: Optional[int]
    clock_unit: Optional[str]


class ConnectMessage(BaseModel):
    worker_version: str
    capabilities: list[str]
    pubkey: str
    slug: str = ""
    sig: str = ""
    ln_url: str  # sent for back compat.  will drop this eventually
    ln_address: str
    auth_key: str  # user private auth token for queenbee
    cpu_count: int
    disk_space: int
    vram: int
    nv_gpu_count: Optional[int] = None
    nv_driver_version: Optional[str] = None
    nv_gpus: Optional[List[GpuInfo]] = []
    cl_driver_version: Optional[str] = None
    cl_gpus: Optional[List[GpuInfo]] = []
    web_gpus: Optional[List[GpuInfo]] = []
    models: List[str] = []


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX, case_sensitive=False)
    auth_key: str = Field('', description="authentication key for a user account")
    queen_url: str = Field(DEFAULT_COORDINATOR, description="websocket url of the coordinator")
    ln_address: str = Field('DONT_PAY_ME', description="a lightning address")
    loops: int = Field(0, description=argparse.SUPPRESS, exclude=True)
    debug: bool = Field(False, description="verbose debugging info")
    test_model: str = Field("", description="specify a HF_REPO/PATH[:FILTER?] to test")
    test_max_tokens: int = Field(16, description="number of tokens to generate when testing")
    low_vram: bool = Field(False, description="use if you have more gpu ram than cpu ram")
    main_gpu: int = 0
    tensor_split: str = Field("", description="comma-delimited list of ratio numbers, one for each gpu")
    force_layers: int = Field(0, description="force layers to load in the gpu")
    layer_offset: int = Field(4, description="reduce the layer guess by this")
    tmp_dir: str = Field(os.path.join(tempfile.gettempdir(), "gputopia-worker"),
                         description="temp folder for data files and checkpoints")

    config: str = Field(os.path.expanduser("~/.config/gputopia"), description="config file location")
    enable: list[Literal["sdxl"]] = Field([], description="List of optional models to enable")
    privkey: str = Field("", description=argparse.SUPPRESS, exclude=True)

class ImageRequest(BaseModel):
    prompt: str
    size: str = "1024x1024"
    n: int = 1
    hyperparameters: dict[str, Any] = {}


class ImageData(BaseModel):
    b64_json: str
    revised_prompt: Optional[str] = None


class ImageResponse(BaseModel):
    created: int
    data: List[ImageData]


def get_free_space_mb(dirname):
    """Return folder/drive free space (in megabytes)."""
    if platform.system() == 'Windows':
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            ctypes.c_wchar_p(dirname), None, None, ctypes.pointer(free_bytes))
        return free_bytes.value / 1024 / 1024
    else:
        st = os.statvfs(dirname)
        return st.f_bavail * st.f_frsize / 1024 / 1024


class WorkerMain:
    def __init__(self, conf: Config):
        self.conn: Optional[websockets.WebSocketClientProtocol] = None
        self.__connect_info: Optional[ConnectMessage] = None
        self.conf = conf
        self._gen_or_load_priv()
        self.__sk = PrivateKey(b64decode(self.conf.privkey))
        self.pubkey = self.__sk.public_key.to_b64()
        if self.conf.main_gpu or self.conf.tensor_split:
            self.slug = b64encode(md5((str(self.conf.main_gpu) + self.conf.tensor_split).encode()).digest()).decode()
        else:
            self.slug = ""
        self.stopped = False
        self.llama = None
        self.whisper = None
        self.llama_model = None
        self.whisper_model = None
        self.llama_cli: Optional[AsyncClient] = None
        self.whisper_cli: Optional[AsyncClient] = None

        if FineTuner:
            self.fine_tuner = FineTuner(self.conf)
        else:
            self.fine_tuner = None
        self.sdxl = SDXL(self.conf)
        self.fast_embed = FastEmbed(self.conf)
        
    def _gen_or_load_priv(self) -> None:
        if not self.conf.privkey:
            cfg = self.conf.config
            if os.path.exists(cfg):
                with open(cfg, encoding="utf8") as fh:
                    js = json.load(fh)
            else:
                js = {}
            if not js.get("privkey"):
                js["privkey"] = b64encode(os.urandom(32)).decode()
                os.makedirs(os.path.dirname(cfg), exist_ok=True)
                with open(cfg, "w", encoding="utf8") as fh:
                    json.dump(js, fh, indent=4)
            self.conf.privkey = js["privkey"]

    def sign(self, msg: ConnectMessage):
        js = msg.model_dump(mode="json")
        js.pop("sig", None)
        # this is needed for a consistent dump!
        dump = json.dumps(js, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
        h32 = sha256(dump.encode()).digest()
        msg.sig = self.__sk.sign(h32)

    async def test_model(self):
        pprint(self.connect_info().model_dump())
        start = time.monotonic()
        await self.load_model(self.conf.test_model)
        load = time.monotonic() - start
        openai_url = "/v1/chat/completions"

        results = []
        for genre in ("sci-fi", "romance", "political", "kids", "teen", "anime"):
            start = time.monotonic()
            openai_req = dict(
                model=self.conf.test_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",
                     "content": f"In the style of Edgar Allen Poe, please write a short {genre} story that is no more than 3 sentences."},
                ],
                max_tokens=self.conf.test_max_tokens
            )
            res: Response = await self.llama_cli.post(openai_url, json=openai_req)
            results.append((res.text, time.monotonic() - start))

        print("Load time:", load)
        sumt = 0.0
        for ent in results:
            usage = json.loads(ent[0])["usage"]
            secs = ent[1]
            sumt += secs
            print("Usage:", usage, secs)
        print("Average:", sumt / len(results))

    async def run(self):
        if self.conf.test_model:
            await self.test_model()
            return

        # in the background, download the model
        if self.sdxl:
            schedule_task(self.sdxl.preload())

        await self.run_ws()

    async def guess_layers(self, model_path):
        if self.conf.force_layers:
            return self.conf.force_layers

        rd = GGUFReader(model_path)

        layers = rd.layers()
        est_ram = rd.vram_estimate()

        # leave room for context
        est_ram += 500000000

        info = self.connect_info()

        tot_mem = 0

        for gpu in info.nv_gpus:
            tot_mem += gpu.memory * 1000000

        if tot_mem == 0:
            for gpu in info.cl_gpus:
                tot_mem += gpu.memory * 1000000

        if est_ram > tot_mem:
            est_layers = tot_mem // (est_ram / layers)
        else:
            est_layers = layers

        log.info("guessing layers: %s (tm %s el %s er %s)",
                 est_layers, tot_mem, est_layers, est_ram)

        return max(0, est_layers - self.conf.layer_offset)

    def clear_llama_model(self):
        if llama_cpp.server.app.llama:
            # critical... must del this before creating a new app
            llama_cpp.server.app.llama = None

        self.llama = None
        self.llama_cli = None
        self.llama_model = None

    def clear_whisper_model(self):
        if whisper_cpp_python.server.app.whisper:
            # critical... must del this before creating a new app
            whisper_cpp_python.server.app.whisper = None

        self.whisper = None
        self.whisper_cli = None
        self.whisper_model = None

    async def load_whisper_model(self, name):
        assert name, "No model name"
        if name == self.whisper_model:
            return
        log.debug("loading model: %s", name)

        model_path = await self.get_model(name, engine="whisper")
        whisper_settings = WhisperSettings(model=model_path)
        self.whisper = create_whisper_app(whisper_settings)
        assert self.whisper, "Load whisper failed.   Try lowering layers."
        self.whisper_cli = AsyncClient(app=self.whisper, base_url="http://test")
        self.whisper_model = name

    async def load_model(self, name):
        assert name, "No model name"
        if name == self.llama_model:
            return

        log.debug("loading model: %s", name)

        model_path = await self.get_model(name)

        sp = None
        if self.conf.tensor_split:
            sp = [float(x) for x in self.conf.tensor_split.split(",")]
        
        self.clear_llama_model()
        if self.sdxl:
            self.sdxl.unload()
 
        settings = LlamaSettings(model=model_path, n_gpu_layers=await self.guess_layers(model_path), seed=-1,
                                 embedding=True, cache=True, port=8181,
                                 main_gpu=self.conf.main_gpu, tensor_split=sp)
        self.llama = create_llama_app(settings)
        assert self.llama, "Load llama failed.   Try lowering layers."
        self.llama_cli = AsyncClient(app=self.llama, base_url="http://test")
        self.llama_model = name
       

    def _get_connect_info(self) -> ConnectMessage:
        disk_space = get_free_space_mb(".")

        caps = []

        caps += ['llama-infer']
        caps += ["whisper"]

        if self.fine_tuner:
            caps += ["llama-fine-tune"]

        if self.fast_embed:
            caps += ["fast-embed"]
        
        if self.sdxl:
            caps += ["sdxl"]

        model_list = self.get_model_list()

        connect_msg = ConnectMessage(
            worker_version=VERSION,
            capabilities=caps,
            ln_url=self.conf.ln_address,  # todo: remove eventually
            pubkey=self.pubkey,
            slug=self.slug,
            ln_address=self.conf.ln_address,
            auth_key=self.conf.auth_key,
            disk_space=int(disk_space),
            cpu_count=multiprocessing.cpu_count(),
            vram=psutil.virtual_memory().available,
            models=model_list
        )

        self.sign(connect_msg)

        try:
            nv = nvidia_smi.getInstance()
            dq = nv.DeviceQuery()

            connect_msg.nv_gpu_count = dq.get("count")
            connect_msg.nv_driver_version = dq["driver_version"]
            connect_msg.nv_gpus = [
                GpuInfo(
                    name=g.get("product_name"),
                    uuid=g.get("uuid"),
                    memory=g.get("fb_memory_usage", {}).get("total"),
                    clock=g.get("clocks", {}).get("graphics_clock"),
                    clock_unit=g.get("clocks", {}).get("unit"),
                ) for g in dq.get("gpu", [])
            ]

        except Exception as ex:
            log.debug("no nvidia: %s", ex)

        try:
            for platf in pyopencl.get_platforms():
                if "nvidia" in platf.name.lower():
                    continue
                connect_msg.cl_driver_version = platf.version
                for device in platf.get_devices():
                    inf = GpuInfo(
                        name=device.name,
                        memory=int(device.global_mem_size / 1000000),
                        clock=device.max_clock_frequency,
                        clock_unit="mhz"
                    )
                    connect_msg.cl_gpus.append(inf)
        except Exception as ex:
            log.debug("no opencl: %s", ex)

        return connect_msg

    def connect_info(self) -> ConnectMessage:
        if not self.__connect_info:
            self.__connect_info = self._get_connect_info()
        return self.__connect_info

    def connect_message(self) -> str:
        info = self.connect_info()
        return info.model_dump_json()

    async def ws_conn(self):
        if not self.conn:
            self.conn = await websockets.connect(self.conf.queen_url, ping_interval=10, ping_timeout=120)
            msg = self.connect_message()
            log.info("connect queen: %s", msg)
            await self.conn.send(msg)

    async def ws_send(self, msg, retry=False):
        await self.ws_conn()
        try:
            return await self.conn.send(msg)
        except (
        websockets.ConnectionClosedError, websockets.ConnectionClosed, websockets.exceptions.ConnectionClosedError):
            self.conn = None
            raise

    async def ws_recv(self):
        await self.ws_conn()
        try:
            return await self.conn.recv()
        except (
        websockets.ConnectionClosedError, websockets.ConnectionClosed, websockets.exceptions.ConnectionClosedError):
            self.conn = None
            raise

    async def run_ws(self):
        loops = 0
        while not self.stopped:
            try:
                await self.run_one()
            finally:
                loops += 1
                if self.conf.loops and loops == self.conf.loops:
                    await asyncio.sleep(1)
                    self.stopped = True

    async def download_tmp_file(self, url: str) -> str:
        import urllib.parse
        res = urllib.parse.urlparse(self.conf.queen_url)
        scheme = "https" if res.scheme == "wss" else "http"
        async with AsyncClient() as cli:
            async with cli.stream('GET', f"{scheme}://{res.netloc}/tmpfile/?filename={url}") as response:
                with tempfile.NamedTemporaryFile("wb", delete=False) as download_file:
                    async for chunk in response.aiter_bytes():
                        download_file.write(chunk)
                    download_file.close()
                return download_file.name
                

    async def run_one(self):
        event = None
        req_str = None
        try:
            req_str = await self.ws_recv()
            req = Req.model_validate_json(req_str)
            model = req.openai_req.get("model")
            log.debug("loading %s", model)

            st = time.monotonic()
            if req.openai_url == "/v1/fine_tuning/jobs":
                self.clear_llama_model()
                if self.sdxl:
                    self.sdxl.unload()
                async for event in self.fine_tuner.fine_tune(req.openai_req):
                    await self.ws_send(json.dumps(event), True)
                await self.ws_send("{}")
            elif req.openai_url == "/v1/images/generations":
                await self.handle_image_generation(req.openai_req)
            elif req.openai_url == "/v1/embeddings" and model.startswith(MODEL_PREFIX):
                res = self.fast_embed.embed(req.openai_req)
                await self.ws_send(json.dumps(res), True)
            elif req.openai_url == "/v1/audio/transcriptions":
                await self.load_whisper_model(model)
                filename = await self.download_tmp_file(req.openai_req["file"])
                file = open(filename, "rb")
                res = await self.whisper_cli.post(req.openai_url, files={"file": file.read()},data={
                    "language": "es",
                    "model": model})
                print("asdasd")
                print(res.text)
                await self.ws_send(json.dumps(res.json()), True)
            elif req.openai_req.get("stream"):
                await self.load_model(model)
                async with aconnect_sse(self.llama_cli, "POST", req.openai_url, json=req.openai_req) as sse:
                    async for event in sse.aiter_sse():
                        if event.data != "[DONE]":
                            await self.ws_send(event.data, True)
                await self.ws_send("{}")
            else:
                await self.load_model(model)
                res: Response = await self.llama_cli.post(req.openai_url, json=req.openai_req)
                await self.ws_send(res.text)
            en = time.monotonic()
            log.info("done %s (%s secs)", model, en - st)
        except (
        websockets.ConnectionClosedError, websockets.ConnectionClosed, websockets.exceptions.ConnectionClosedError):
            if req_str:
                log.error("disconnected while running request: %s", req_str)
            if event:
                log.error("was sending event: %s", event)
        except Exception as ex:
            log.exception("error running request: %s", req_str)
            try:
                if self.conn:
                    await self.ws_send(json.dumps({"error": str(ex), "error_type": type(ex).__name__}), True)
            except Exception as ex:
                log.exception("error reporting error: %s", str(ex))

    async def handle_image_generation(self, request_data):
        self.clear_llama_model()
        res = await self.sdxl.handle_req(request_data)
        await self.ws_send(json.dumps(res), True)

    async def get_model(self, name, engine="llama"):
        if engine == "llama":
            return await self.download_model(name)
        elif engine == "whisper":
            return await self.download_whisper_model(name)

    async def download_file(self, url: str) -> str:
        output_file = url_to_tempfile(self.conf, url)
        if not os.path.exists(output_file):
            with open(output_file + ".tmp", "wb") as fh:
                async with AsyncClient() as cli:
                    async with cli.stream("GET", url) as res:
                        res: Response
                        async for chunk in res.aiter_bytes():
                            fh.write(chunk)
            os.replace(output_file + ".tmp", output_file)
        return output_file

    def note_have(self, name: str):  # noqa
        mods = self.get_model_info_from_config()
        mods[name] = {"time": time.time()}
        self.write_model_info_to_config(mods)

    def note_dropped(self, name: str):  # noqa
        mods = self.get_model_info_from_config()
        mods.pop(name, None)
        self.write_model_info_to_config(mods)

    def write_model_info_to_config(self, mods):
        cfg_path = self.conf.config
        with open(cfg_path + ".models.tmp", "w") as fh:
            json.dump(mods, fh)
        os.replace(cfg_path + ".models.tmp", cfg_path + ".models")

    def get_model_info_from_config(self) -> dict[str]:
        cfg_path = self.conf.config
        try:
            with open(cfg_path + ".models") as fh:
                mods = json.load(fh)
        except (json.JSONDecodeError, FileNotFoundError):
            mods = {}
        return mods

    def check_have_url_model(self, name: str):
        if name.startswith(USER_PREFIX):
            name = user_ft_name_to_url(name)
        if not name.startswith("https:"):
            return False
        output_file = url_to_tempfile(self.conf, name)
        if not os.path.exists(output_file):
            return False
        return True

    async def download_whisper_model(self, name):
        # uses hf cache, so no need to handle here
        orig_name = name

        if name.startswith(USER_PREFIX):
            name = user_ft_name_to_url(name)

        if name.startswith("https:"):
            ret = await self.download_file(name)
            self.note_have(orig_name)
            return ret
        from ai_worker.ggml import download_ggml
        # size = get_size(name)
        # await self.free_up_space(size)
        loop = asyncio.get_running_loop()
        path = await loop.run_in_executor(None, lambda: download_ggml(name))
        self.note_have(orig_name)

        return path

    async def download_model(self, name):
        # uses hf cache, so no need to handle here
        orig_name = name

        if name.startswith(USER_PREFIX):
            name = user_ft_name_to_url(name)

        if name.startswith("https:"):
            ret = await self.download_file(name)
            self.note_have(orig_name)
            return ret

        from gguf_loader.main import download_gguf
        size = get_size(name)
        await self.free_up_space(size)
        loop = asyncio.get_running_loop()
        path = await loop.run_in_executor(None, lambda: download_gguf(name))
        self.note_have(orig_name)

        return path

    def get_model_list(self) -> list[str]:
        dct = self.get_model_info_from_config()

        # search hf caceh
        from gguf_loader.main import get_model_list

        all_cached = set()
        for ent in get_model_list():
            all_cached.add(ent)
            if ent not in dct:
                # stuff in the config, if we never saw it before
                self.note_have(ent)

        # prune list to ones you really have
        for k in list(dct.keys()):
            if k not in all_cached and not self.check_have_url_model(k):
                self.note_dropped(k)

        dct = self.get_model_info_from_config()

        # force loaded to be first/pref
        if self.llama_model:
            dct[self.llama_model] = {"time": time.time()}

        # most recent first, which is always the loaded one
        return sorted(dct.keys(), key=lambda k: dct[k].get("time"), reverse=True)

    @staticmethod
    def report_pct(name, pct):
        print("\r", name, pct, end='')

    async def free_up_space(self, size):
        pass


def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        stream=sys.stdout)

    parser = argparse.ArgumentParser(description=f"""
Start an AI worker and start accepting jobs.

At a minimum, specify an --ln_address=xxxx@getalby.com.

You can specify any config variables in a .env or via the environment variable instead.
Use {ENV_PREFIX} as the prefix.

For example:
{ENV_PREFIX}_LN_ADDRESS=xxxx@getalby.com 
    """, formatter_class=argparse.RawTextHelpFormatter)

    arg_names = []
    for name, field in Config.model_fields.items():
        description = field.description
        if field.default and description is not None and description != argparse.SUPPRESS:
            description += f" (default: {field.default})"
        args = dict(
            dest=name,
            type=field.annotation if field.annotation is not None else str,
            help=description,
            action="store_true" if field.annotation is bool else "store",
        )
        if field.default:
            args["default"] = field.default
        if field.annotation is bool:
            args.pop("type")
        if name == "enable":
            continue
        arg_names.append(name)
        parser.add_argument(f"--{name}", **args)

    parser.add_argument("--version", action="store_true")

    # todo: back compat.   remove eventually
    parser.add_argument("--ln_url", type=str, help=argparse.SUPPRESS)
    

    # too annoying to deal with Listeral list
    parser.add_argument("--enable", choices=["sdxl"], nargs="+")
    arg_names.append("enable")

    args = parser.parse_args(args=argv)

    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf8") as fh:
            for k, v in json.load(fh).items():
                cv = getattr(args, k)
                if cv is None or cv == Config.model_fields[k].default:
                    setattr(args, k, v)

    if args.debug:
        log.setLevel(logging.DEBUG)

    if args.ln_url:
        args.ln_address = args.ln_url

    if args.version:
        print(VERSION)
        exit(0)

    if os.path.exists("gputopia-worker.ini"):
        logging.config.fileConfig("gputopia-worker.ini")

    conf = Config(**{k: getattr(args, k) for k in arg_names if getattr(args, k) is not None})

    log.debug("config: %s", conf)
    wm = WorkerMain(conf)

    asyncio.run(wm.run())
