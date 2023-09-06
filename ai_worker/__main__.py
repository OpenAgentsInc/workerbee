import os
import argparse

import uvicorn

APP_NAME="gputopia"
DEFAULT_COORDINATOR = "https://gputopia.ai/api/v1"
DEFAULT_BASE = "https://gputopia.ai/models"

from llama_cpp.server.app import create_app as create_llama_app, Settings as LlamaSettings
from pydantic_settings import BaseConfig, SettingsConfigDict
from pydantic import BaseModel

class Req(BaseModel):
    openai_url: str
    openai_req: dict
    

class Config(BaseConfig):
    model_config = SettingsConfigDict(env_prefix=APP_NAME +'_worker', case_sensitive=False)
    auth_key: str = ""
    coordinator_url: str = DEFAULT_COORDIANTOR
    model_base_url: str = DEFAULT_BASE_URL
    model_dir: str = os.path.expanduser('~/.ai-models')

class WorkerMain:
    def __init__(self, conf: Config):
        self.conf = conf
        self.stopped = False
        self.llama = None
        self.llama_model = None

    async def run(self):
        while True:
            async for websocket in websockets.connect(self.conf.coordinator_url):
                try:
                    await self.run_ws(websocket)
                except websockets.ConnectionClosed:
                    continue

    
    def load_model(self, name):
        if name == self.llama_model:
            return
        model_path = await self.get_model(model)
        settings = LlamaSettings(model=model_path, n_gpu_layers=self.guess_layers(model_path), seed=-1, embedding=True, cache=True, port=8181)
        self.llama = create_app(settings)

    
    async def run_ws(self, ws):
        await ws.send(self.connect_message())
        
        while not self.stopped:
            req_str = await ws.recv()
            req = Req.from_json(req_str)
            model = req.openai_req.get("model")

            self.load_model(model)

            if req.openai_url.endswith("embeddings"):
                request = CreateEmbeddingRequest.model_validate(req.openai_req)
            else:
                request = CreateChatCompletionRequest.model_validate(req.openai_req)
           
            create_chat_completion

    async def get_model(self, name):
        ret = self.get_local_model(name)
        if ret:
            return ret
        return await self.download_model(name)

    def get_local_model(name):
        dest = self.model_file_for(name)
        if os.path.getsize(dest) > 0:
            return dest
        return None

    def model_file_for(name):
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

            done += 0
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


if __name__ == "__main__":

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
