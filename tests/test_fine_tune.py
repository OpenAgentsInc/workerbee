import os

import pytest

from ai_worker.fine_tune import FineTuner
from ai_worker.main import Config


@pytest.fixture
def ft():
    conf = Config()
    ft = FineTuner(conf)
    yield ft


async def test_dl(ft):
    fil = await ft.download_file("https://gputopia-user-bucket.s3.amazonaws.com/bypass/file_782")
    assert os.path.getsize(fil) == 782


async def test_basic(ft):
    job = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "training_file": "https://gputopia-user-bucket.s3.amazonaws.com/bypass/file_782",
    }
    fin = []
    async for res in ft.fine_tune(job):
        fin.append(res)

    assert fin[-1]["status"] == "done"
