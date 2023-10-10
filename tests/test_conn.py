import asyncio
import json
import re
import time
from threading import Thread
from typing import Any, Optional

import pytest
import websockets

from ai_worker.key import PublicKey
from ai_worker.main import WorkerMain, Config, main as worker_main
from gguf_loader.main import download_gguf, main as loader_main, get_size

from pynvml.smi import nvidia_smi
from pynvml.nvml import NVMLError

queen_events = []


async def queen(websocket, _path):
    model = "TheBloke/WizardLM-7B-uncensored-GGML:q4_K_M"

    async for message in websocket:
        queen_events.append(message)
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
    srv = websockets.serve(queen, "127.0.0.1", 0)
    res = loop.run_until_complete(srv)
    ret[0] = res
    ret[1] = loop
    loop.run_forever()


@pytest.fixture(scope="module")
def test_queen():
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


def test_conn_str(tmp_path):
    wm = WorkerMain(Config(config=str(tmp_path / "tmp")))
    msg = wm.connect_message()
    js = json.loads(msg)

    sig = js.pop("sig")
    orig = json.dumps(js, ensure_ascii=False, sort_keys=True)

    pub = PublicKey.from_hex(wm.pubkey)
    pub.verify(sig, orig.encode())

    wm2 = WorkerMain(Config(config=str(tmp_path / "tmp")))
    assert wm2.privkey == wm.privkey

    try:
        inst = nvidia_smi.getInstance()
        dq = inst.DeviceQuery()
        # sometimes it throws an error... sometimes not!
        if dq.get("count"):
            assert js["nv_driver_version"]
            assert js["nv_gpu_count"]
    except NVMLError:
        pass

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


async def test_run(test_queen):
    queen_events.clear()
    wm = WorkerMain(Config(loops=1, queen_url=test_queen))
    await wm.run()
    while len(queen_events) == 1:
        time.sleep(0.2)
    assert len(queen_events) == 2
    js = json.loads(queen_events[1])
    assert not js.get("error")


def test_download_model():
    assert get_size("TheBloke/WizardLM-7B-uncensored-GGML:q4_K_M") > 0
    assert download_gguf("TheBloke/WizardLM-7B-uncensored-GGML:q4_K_M")


def test_download_main(capsys):
    loader_main(["TheBloke/WizardLM-7B-uncensored-GGML:q4_K_M"])
    oe = capsys.readouterr().out
    assert "q4_K_M" in oe


def test_version(capsys):
    try:
        worker_main(["--version"])
    except SystemExit:
        pass
    oe = capsys.readouterr().out
    re.match(r"\d+\.\d+\.\d+", oe)


def test_main(capsys):
    try:
        worker_main(["--version"])
        assert False, "should exit on --version call"
    except SystemExit:
        pass
    oe = capsys.readouterr().out
    re.match(r"\d+\.\d+\.\d+", oe)


def test_cfg(capsys, tmp_path):
    with open(tmp_path / "tmp", "w") as fh:
        json.dump(dict(debug=True, test_model="TheBloke/CodeLlama-7B-Instruct-GGUF:Q4_K_M", test_max_tokens=1),
                  fh)
    worker_main(["--config", str(tmp_path / "tmp")])

    oe = capsys.readouterr().out
    # log shows pubkey and total_tokens because debug and test_models are set
    assert re.search(r"pubkey", oe)
    assert re.search(r"total_tokens", oe)
