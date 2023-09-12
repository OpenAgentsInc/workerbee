import asyncio
import json
import time
from threading import Thread
from typing import Any, Optional

import pytest
import websockets
from ai_worker.main import WorkerMain, Config
from gguf_loader.main import download_gguf, main as loader_main, get_size

try:
    from pynvml.smi import nvidia_smi
except ImportError:
    nvidia_smi = None

spider_events = []


async def spider(websocket, _path):
    model = "TheBloke/WizardLM-7B-uncensored-GGML:q4_K_M"

    async for message in websocket:
        spider_events.append(message)
        if message:
            jj = json.loads(message)
            if jj.get("cpu_count"):
                data = {"openai_url": "/v1/chat/completions", "openai_req": dict(
                    model=model,
                    stream=False,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": "hello"},
                    ],
                )}
                await websocket.send(json.dumps(data))


def start_server(ret):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    srv = websockets.serve(spider, "127.0.0.1", 0)
    res = loop.run_until_complete(srv)
    ret[0] = res
    ret[1] = loop
    loop.run_forever()


@pytest.fixture(scope="module")
def test_spider():
    ret: list[Any] = [None, None]
    thread = Thread(target=start_server, args=(ret,), daemon=True)
    thread.daemon = True
    thread.start()
    while not ret[0]:
        time.sleep(0.1)
    server = ret[0]
    port = server.sockets[0].getsockname()[1]
    yield f"ws://127.0.0.1:{port}"
    server.close()
    # time to close
    time.sleep(0.3)
    ret[1].stop()


def test_conn_str():
    wm = WorkerMain(Config())
    msg = wm.connect_message()
    js = json.loads(msg)

    if nvidia_smi:
        assert js["nv_driver_version"]
        assert js["nv_gpu_count"]

    assert js["cpu_count"]
    assert js["vram"]


async def test_wm():
    wm = WorkerMain(Config())
    await wm.load_model("TheBloke/WizardLM-7B-uncensored-GGML:q4_K_M")
    res = wm.llama_cli.post("/v1/chat/completions", json=dict(
        model=wm.llama_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "hello"},
        ]
    ))
    assert res


async def test_run(test_spider):
    spider_events.clear()
    wm = WorkerMain(Config(once=True, spider_url=test_spider))
    await wm.run()
    while len(spider_events) == 1:
        time.sleep(0.2)
    assert len(spider_events) == 2


def test_download_model():
    assert get_size("TheBloke/WizardLM-7B-uncensored-GGML:q4_K_M") > 0
    assert download_gguf("TheBloke/WizardLM-7B-uncensored-GGML:q4_K_M")


def test_download_main(capsys):
    loader_main(["TheBloke/WizardLM-7B-uncensored-GGML:q4_K_M"])
    oe = capsys.readouterr().out
    assert "q4_K_M" in oe
