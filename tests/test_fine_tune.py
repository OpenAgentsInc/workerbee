import os
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel
import pytest
import tarfile
import tempfile
import logging as log

from ai_worker.fine_tune import FineTuner
from ai_worker.main import Config
from ai_worker.fine_tune import gzip

@pytest.fixture
def ft():
    conf = Config()
    ft = FineTuner(conf)
    yield ft


async def test_dl(ft):
    fil = await ft.download_file("https://gputopia-user-bucket.s3.amazonaws.com/bypass/file_782")
    assert os.path.getsize(fil) == 782


async def test_peft_unload_save_and_gguf(ft):
    base_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    args = {}
    args.update(dict(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ))
    log.info("load peft")

    bnb_config = BitsAndBytesConfig(**args)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.save_pretrained(ft.temp_file("test"))

    model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, local_files_only=True, device_map="auto"), "mistral-journal-finetune")
    
    got = {}
    def cb(state):
        if state.pop("lora_chunk", None):
            got["lora"] += 1
        if state.pop("gguf_chunk", None):
            got["gguf"] += 1
    
    log.info("call rf")

    ft.return_final("test", model, cb)

    log.debug(got)

    assert got["lora"]
    assert got["gguf"]


async def test_e2e(ft):
    job = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "training_file": "https://gputopia-user-bucket.s3.amazonaws.com/bypass/file_782",
    }
    fin = []
    got = {}
    async for res in ft.fine_tune(job):
        if res.pop("chunk", None):
            if res.get("lora"):
                got["lora"] += 1
            elif res.get("gguf"):
                got["gguf"] += 1
        log.info("test-result: %s", res)
        fin.append(res)

    log.debug(got)
    assert got["lora"]
    assert got["gguf"]
    assert fin[-1]["status"] == "done"


def test_gzip():
    with tempfile.TemporaryDirectory() as tmpdirname:
        test_folder = os.path.join(tmpdirname, "test_folder")
        os.makedirs(test_folder)
        
        with open(os.path.join(test_folder, "test_file.txt"), "w") as f:
            f.write("This is a test file.")
        
        gzip(test_folder)
        
        assert os.path.exists(test_folder + ".tar.gz")
        
        with tarfile.open(test_folder + ".tar.gz", 'r:gz') as archive:
            assert "test_folder/test_file.txt" in archive.getnames()


