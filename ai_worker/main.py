import argparse
import asyncio
import json
import multiprocessing
from typing import Optional
import logging as log

import psutil
import sseclient
import websockets
from llama_cpp.server.app import Settings as LlamaSettings, create_app as create_llama_app
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from pynvml.smi import nvidia_smi
from fastapi.testclient import TestClient
from starlette.responses import Response

from gguf_loader.main import get_size

APP_NAME= "gputopia"
DEFAULT_COORDINATOR = "wss://gputopia.ai/api/v1"


class Req(BaseModel):
    openai_url: str
    openai_req: dict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=APP_NAME +'_worker', case_sensitive=False)
    auth_key: str = ""
    coordinator_url: str = DEFAULT_COORDINATOR
    ln_url: str = "DONT_PAY_ME"


class WorkerMain:
    def __init__(self, conf: Config):
        self.conf = conf
        self.stopped = False
        self.llama = None
        self.llama_model = None
        self.llama_cli: Optional[TestClient] = None

    async def run(self):
        while True:
            async for websocket in websockets.connect(self.conf.coordinator_url):
                try:
                    await self.run_ws(websocket)
                except websockets.ConnectionClosed:
                    continue

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
            ret.update(dict(
                nv_gpu_count=dq.get("count"),
                nv_driver_version=dq.get("driver_version"),
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

    async def run_ws(self, ws):
        await ws.send(self.connect_message())

        while not self.stopped:
            req_str = await ws.recv()
            req = Req.from_json(req_str)
            model = req.openai_req.get("model")

            await self.load_model(model)

            res: Response = self.llama_cli.post(req.openai_url, json=req.openai_req)

            if res.headers.get('Content-Type') == 'text/event-stream':
                sse = sseclient.SSEClient(res)

                for event in sse:
                    ws.send(event.data)
                ws.send("")
            else:
                ws.send(res.body.decode("urf-8"))

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
