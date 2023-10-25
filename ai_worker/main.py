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
from typing import Optional, List
from base64 import urlsafe_b64encode as b64encode, urlsafe_b64decode as b64decode
import psutil
import websockets
from httpx import Response, AsyncClient
from httpx_sse import aconnect_sse
from llama_cpp.server.app import Settings as LlamaSettings, create_app as create_llama_app
import llama_cpp.server.app
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pynvml.smi import nvidia_smi
import pyopencl
from dotenv import load_dotenv

try:
    from .fine_tune import FineTuner
except ImportError as ex:
    log.error("failed import, disabling fine tune: %s", repr(ex))
    FineTuner = None

from gguf_loader.main import get_size

from .gguf_reader import GGUFReader
from .key import PrivateKey
from .version import VERSION

APP_NAME = "gputopia"
ENV_PREFIX = APP_NAME.upper()

DEFAULT_COORDINATOR = "wss://queenbee.gputopia.ai/worker"

log = logging.getLogger(__name__)

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
    layer_offset: int = Field(2, description="reduce the layer guess by this")
    tmp_dir: str = Field(os.path.join(tempfile.gettempdir(), "gputopia-worker"),
                         description="temp folder for data files and checkpoints")

    config: str = Field(os.path.expanduser("~/.config/gputopia"), description="config file location")
    privkey: str = Field("", description=argparse.SUPPRESS, exclude=True)


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
        self.llama_model = None
        self.llama_cli: Optional[AsyncClient] = None
        if FineTuner:
            self.fine_tuner = FineTuner(self.conf)

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

        await self.run_ws()

    async def guess_layers(self, model_path):
        if self.conf.force_layers:
            return self.conf.force_layers

        rd = GGUFReader(model_path)

        layers = rd.layers()
        est_ram = rd.vram_estimate()

        info = self.connect_info()

        tot_mem = 0

        for gpu in info.nv_gpus:
            tot_mem += gpu.memory * 1000000

        if est_ram > tot_mem:
            est_layers = tot_mem // (est_ram / layers)
        else:
            est_layers = layers

        log.info("guessing layers: %s (tm %s el %s er %s)",
                 est_layers, tot_mem, est_layers, est_ram)

        return max(0, est_layers - self.conf.layer_offset)

    async def load_model(self, name):
        if name == self.llama_model:
            return
        log.debug("loading model: %s", name)
        model_path = await self.get_model(name)

        if llama_cpp.server.app.llama:
            # critical... must del this before creating a new app
            del llama_cpp.server.app.llama

        sp = None
        if self.conf.tensor_split:
            sp = [float(x) for x in self.conf.tensor_split.split(",")]
        settings = LlamaSettings(model=model_path, n_gpu_layers=await self.guess_layers(model_path), seed=-1,
                                 embedding=True, cache=True, port=8181,
                                 main_gpu=self.conf.main_gpu, tensor_split=sp)
        self.llama = create_llama_app(settings)
        self.llama_cli = AsyncClient(app=self.llama, base_url="http://test")
        self.llama_model = name

    def _get_connect_info(self) -> ConnectMessage:
        disk_space = get_free_space_mb(".")

        caps = []

        caps += ['llama-infer']

        if self.fine_tuner:
            caps += ["llama-fine-tune"]

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
        except (websockets.ConnectionClosedError, websockets.ConnectionClosed):
            self.conn = None
            raise

    async def ws_recv(self):
        await self.ws_conn()
        try:
            return await self.conn.recv()
        except (websockets.ConnectionClosedError, websockets.ConnectionClosed):
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

    async def run_one(self):
        req_str = await self.ws_recv()
        event = None
        try:
            req = Req.model_validate_json(req_str)
            model = req.openai_req.get("model")

            log.debug("loading %s", model)

            st = time.monotonic()
            if req.openai_url == "/v1/fine_tuning/jobs":
                async for event in self.fine_tuner.fine_tune(req.openai_req):
                    await self.ws_send(json.dumps(event), True)
                await self.ws_send("{}")
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
        except (websockets.ConnectionClosedError, websockets.ConnectionClosed):
            log.error("disconnected while running request: %s", req_str)
            if event:
                log.error("was sending event: %s", event)
        except Exception as ex:
            log.exception("error running request: %s", req_str)
            await self.ws_send(json.dumps({"error": str(ex), "error_type": type(ex).__name__}), True)

    async def get_model(self, name):
        return await self.download_model(name)

    async def download_file(self, url: str) -> str:
        name = hashlib.md5(url.encode()).hexdigest()
        output_file = os.path.join(self.conf.tmp_dir, name)
        if not os.path.exists(output_file):
            with open(output_file + ".tmp", "wb") as fh:
                async with AsyncClient() as cli:
                    res: Response = await cli.get(url)
                    async for chunk in res.aiter_bytes():
                        fh.write(chunk)
            os.replace(output_file + ".tmp", output_file)
        return output_file

    async def download_model(self, name):
        # uses hf cache, so no need to handle here
        user_prefix = "user:"
        
        if name.startswith(user_prefix):
            sub = name[user_prefix:]
            if not sub.endswith(".gguf"):
                sub = sub + ".gguf"
            name = f"https://gputopia-user-bucket.s3.amazonaws.com/{sub}"

        if name.startswith("https:"):
            return await self.download_file(name)
 
        from gguf_loader.main import download_gguf
        size = get_size(name)
        await self.free_up_space(size)
        loop = asyncio.get_running_loop()
        path = await loop.run_in_executor(None, lambda: download_gguf(name))
        return path

    def report_done(self, name):
        print("\r", name, 100)

    def report_pct(self, name, pct):
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
        arg_names.append(name)
        parser.add_argument(f"--{name}", **args)

    parser.add_argument("--version", action="store_true")

    # todo: back compat.   remove eventually
    parser.add_argument("--ln_url", type=str, help=argparse.SUPPRESS)

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
