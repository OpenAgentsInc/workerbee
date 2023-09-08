import argparse
import asyncio
import json
import os
from typing import Optional

import httpx
import sseclient
import websockets
from llama_cpp.server.app import Settings as LlamaSettings, create_app as create_llama_app
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from pynvml.smi import nvidia_smi
from fastapi.testclient import TestClient
from starlette.responses import Response

APP_NAME= "gputopia"
DEFAULT_COORDINATOR = "https://gputopia.ai/api/v1"
DEFAULT_BASE_URL = "https://gputopia.ai/models"


class Req(BaseModel):
    openai_url: str
    openai_req: dict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=APP_NAME +'_worker', case_sensitive=False)
    auth_key: str = ""
    coordinator_url: str = DEFAULT_COORDINATOR
    model_base_url: str = DEFAULT_BASE_URL
    model_dir: str = os.path.expanduser('~/.ai-models')


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
        return 30

    async def load_model(self, name):
        if name == self.llama_model:
            return
        model_path = await self.get_model(name)
        settings = LlamaSettings(model=model_path, n_gpu_layers=self.guess_layers(model_path), seed=-1, embedding=True, cache=True, port=8181)
        self.llama = create_llama_app(settings)
        self.llama_cli = TestClient(self.llama)

    @staticmethod
    def connect_message():
        nv = nvidia_smi.getInstance()
        dq = nv.DeviceQuery()

        return json.dumps(dict(
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
        ret = self.get_local_model(name)
        if ret:
            return ret
        return await self.download_model(name)

    def get_local_model(self, name):
        dest = self.model_file_for(name)
        if os.path.getsize(dest) > 0:
            return dest
        return None

    def model_file_for(self, name):
        return self.conf.model_dir + "/" + name.replace("/", ".")

    async def download_model(self, name):
        url = self.conf.model_base_url + "/" + name.replace("/", ".")

        async with httpx.AsyncClient() as client:
            r = await client.head(url)
            size = r.headers.get('Content-Length')
            if not size:
                params = self.get_model_params(name)
                bits = self.get_model_bits(name)
                # 70b * 4 bit = 35gb (roughly)
                size = params * bits / 8

            assert size, "unable to estimate model size, not downloading"

            await self.free_up_space(size)

            dest = self.model_file_for(name)

            done = 0
            with open(dest + ".tmp", "wb") as f:
                async with client.stream("GET", url) as r:
                    async for chunk in r.aiter_bytes():
                        f.write(chunk)
                        done += len(chunk)
                        self.report_pct(name, done/size)
            os.replace(dest + ".tmp", dest)
            self.report_done(name)

        return dest

    def report_done(self, name):
        print("\r", name, 100)

    def report_pct(self, name, pct):
        print("\r", name, pct, end='')


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

    asyncio.run(wm.main())
