import argparse
import asyncio
import json
import multiprocessing
from typing import Optional
import logging as log

import psutil
import websockets
from httpx import Response
from httpx_sse import connect_sse
from llama_cpp.server.app import Settings as LlamaSettings, create_app as create_llama_app
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from pynvml.smi import nvidia_smi
from fastapi.testclient import TestClient

from gguf_loader.main import get_size

APP_NAME= "gputopia"
DEFAULT_COORDINATOR = "wss://gputopia.ai/api/v1"


class Req(BaseModel):
    openai_url: str
    openai_req: dict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=APP_NAME +'_worker', case_sensitive=False)
    auth_key: str = ""
    spider_url: str = DEFAULT_COORDINATOR
    ln_url: str = "DONT_PAY_ME"
    once: bool = False


class WorkerMain:
    def __init__(self, conf: Config):
        self.conf = conf
        self.stopped = False
        self.llama = None
        self.llama_model = None
        self.llama_cli: Optional[TestClient] = None

    async def run(self):
        async for websocket in websockets.connect(self.conf.spider_url):
            if self.stopped:
                break
            try:
                await self.run_ws(websocket)
            except websockets.ConnectionClosed:
                continue
            if self.stopped:
                break

    async def guess_layers(self, model_path):
        # todo: read model file and compare to gpu resources
        return 20

    async def load_model(self, name):
        if name == self.llama_model:
            return
        model_path = await self.get_model(name)
        settings = LlamaSettings(model=model_path, n_gpu_layers=await self.guess_layers(model_path), seed=-1, embedding=True, cache=True, port=8181)
        self.llama = create_llama_app(settings)
        self.llama_cli = TestClient(self.llama)

    def connect_message(self) -> str:
        ret = dict(
            ln_url=self.conf.ln_url,
            cpu_count=multiprocessing.cpu_count(),
            vram=psutil.virtual_memory().available
        )

        try:
            # get nvidia info...todo: amd support
            nv = nvidia_smi.getInstance()
            dq = nv.DeviceQuery()
            # force throw error on version not there....
            ret.update(dict(
                nv_gpu_count=dq.get("count"),
                nv_driver_version=dq["driver_version"],
                nv_gpus=[
                    dict(
                        name=g.get("product_name"),
                        uuid=g.get("uuid"),
                        memory=g.get("fb_memory_usage", {}).get("total")
                    ) for g in dq.get("gpu", [])
                ]
            ))
        except Exception as ex:
            log.debug("no nvidia: %s", ex)
            pass

        return json.dumps(ret)

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

            await self.load_model(model)

            if req.openai_req.get("stream"):
                with connect_sse(self.llama_cli, "POST", req.openai_url, json=req.openai_req) as sse:
                    for event in sse.iter_sse():
                        await ws.send(event.data)
                await ws.send("")
            else:
                res: Response = self.llama_cli.post(req.openai_url, json=req.openai_req)
                await ws.send(res.text)
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
    parser = argparse.ArgumentParser()
    for name, field in Config.model_fields.items():
        description = field.description
        if field.default is not None and description is not None:
            description += f" (default: {field.default})"
        parser.add_argument(
            f"--{name}",
            dest=name,
            type=field.annotation if field.annotation is not None else str,
            help=description,
            action="store_true" if field.annotation is bool else "store",
        )

    args = parser.parse_args()

    conf = Config(**{k: v for k, v in vars(args).items() if v is not None})

    wm = WorkerMain(conf)

    asyncio.run(wm.run())
