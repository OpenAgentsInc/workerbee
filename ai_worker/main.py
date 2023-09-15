import argparse
import asyncio
import ctypes
import json
import logging
import multiprocessing
import os
import platform
import time
from pprint import pprint
from typing import Optional, List

import psutil
import websockets
from httpx import Response, AsyncClient
from httpx_sse import aconnect_sse
from llama_cpp.server.app import Settings as LlamaSettings, create_app as create_llama_app
import llama_cpp.server.app
from llama_cpp import llama_free_model, llama_free
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from pynvml.smi import nvidia_smi
import pyopencl

from gguf_loader.main import get_size

from .gguf_reader import GGUFReader

APP_NAME = "gputopia"
DEFAULT_COORDINATOR = "wss://queenbee.gputopia.ai/worker"

log = logging.getLogger(__name__)


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
    ln_url: str
    cpu_count: int
    disk_space: int
    vram: int
    nv_gpu_count: Optional[int] = None
    nv_driver_version: Optional[str] = None
    nv_gpus: Optional[List[GpuInfo]] = []
    cl_driver_version: Optional[str] = None
    cl_gpus: Optional[List[GpuInfo]] = []


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=APP_NAME + '_worker', case_sensitive=False)
    auth_key: str = ""
    spider_url: str = DEFAULT_COORDINATOR
    ln_url: str = "DONT_PAY_ME"
    once: bool = False
    debug: bool = False
    test_model: str = ""
    test_max_tokens: int = 16
    low_vram: bool = False
    force_layers: int = 0


def get_free_space_mb(dirname):
    """Return folder/drive free space (in megabytes)."""
    if platform.system() == 'Windows':
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(dirname), None, None, ctypes.pointer(free_bytes))
        return free_bytes.value / 1024 / 1024
    else:
        st = os.statvfs(dirname)
        return st.f_bavail * st.f_frsize / 1024 / 1024


class WorkerMain:
    def __init__(self, conf: Config):
        self.__connect_info: Optional[ConnectMessage] = None
        self.conf = conf
        self.stopped = False
        self.llama = None
        self.llama_model = None
        self.llama_cli: Optional[AsyncClient] = None

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
                    {"role": "user", "content": f"Write a short {genre} story."},
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

        async for websocket in websockets.connect(self.conf.spider_url):
            if self.stopped:
                break
            try:
                await self.run_ws(websocket)
            except websockets.ConnectionClosed:
                continue
            except Exception:
                log.exception("error in worker")
            if self.stopped:
                break

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

        log.info("guessing layers: %s (tm %s el %s er %s)", est_layers, tot_mem, est_layers, est_ram)

        return est_layers

    async def load_model(self, name):
        if name == self.llama_model:
            return
        log.debug("loading model: %s", name)
        model_path = await self.get_model(name)

        if llama_cpp.server.app.llama:
            # critical... must del this before creating a new app
            del llama_cpp.server.app.llama

        settings = LlamaSettings(model=model_path, n_gpu_layers=await self.guess_layers(model_path), seed=-1,
                                 embedding=True, cache=True, low_vram=self.conf.low_vram, port=8181)
        self.llama = create_llama_app(settings)
        self.llama_cli = AsyncClient(app=self.llama, base_url="http://test")
        self.llama_model = name

    def _get_connect_info(self) -> ConnectMessage:
        disk_space = get_free_space_mb(".")

        connect_msg = ConnectMessage(
            ln_url=self.conf.ln_url,
            auth_key=self.conf.auth_key,
            disk_space=int(disk_space),
            cpu_count=multiprocessing.cpu_count(),
            vram=psutil.virtual_memory().available,
        )

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
            for platform in pyopencl.get_platforms():
                if "nvidia" in platform.name.lower():
                    continue
                connect_msg.cl_driver_version = platform.version
                for device in platform.get_devices():
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

    async def run_ws(self, ws: websockets.WebSocketCommonProtocol):
        await ws.send(self.connect_message())

        while not self.stopped:
            await self.run_one(ws)
            if self.conf.once:
                await asyncio.sleep(1)
                self.stopped = True

    async def run_one(self, ws: websockets.WebSocketCommonProtocol):
        req_str = await ws.recv()
        try:
            req = Req.model_validate_json(req_str)
            model = req.openai_req.get("model")

            log.debug("loading %s", model)

            await self.load_model(model)

            if req.openai_req.get("stream"):
                async with aconnect_sse(self.llama_cli, "POST", req.openai_url, json=req.openai_req) as sse:
                    async for event in sse.aiter_sse():
                        await ws.send(event.data)
                await ws.send("")
            else:
                res: Response = await self.llama_cli.post(req.openai_url, json=req.openai_req)
                await ws.send(res.text)
            
            log.debug("done %s", model)
        except Exception as ex:
            log.exception("error running request: %s", req_str)
            await ws.send(json.dumps({"error": str(ex), "error_type": type(ex).__name__}))

    async def get_model(self, name):
        return await self.download_model(name)

    async def download_model(self, name):
        # uses hf cache, so no need to handle here
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


def main():
    logging.basicConfig()
    log.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    for name, field in Config.model_fields.items():
        description = field.description
        if field.default is not None and description is not None:
            description += f" (default: {field.default})"
        args = dict(
            dest=name,
            type=field.annotation if field.annotation is not None else str,
            help=description,
            action="store_true" if field.annotation is bool else "store",
        )
        if field.annotation is bool:
            args.pop("type")
        parser.add_argument(f"--{name}", **args)

    args = parser.parse_args()
    if args.debug:
        log.setLevel(logging.DEBUG)

    conf = Config(**{k: v for k, v in vars(args).items() if v is not None})

    wm = WorkerMain(conf)

    asyncio.run(wm.run())
