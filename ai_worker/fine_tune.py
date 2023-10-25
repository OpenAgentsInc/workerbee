import gc
import hashlib
import json
import asyncio
import threading
import logging
import os
import random
import tarfile
import shutil

import transformers
from datasets import load_dataset
from httpx import AsyncClient, Response
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback 
from peft import prepare_model_for_kbit_training, PeftModel, LoraConfig, get_peft_model
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

from ai_worker.util import b64enc
from gguf_loader.convert import main as gguf_main

MAX_CONTEXT = 300000

log = logging.getLogger(__name__)

def gzip(folder):
    """tar gz the folder to 'folder.tar.gz', removes the folder"""
    base_folder_name = os.path.basename(folder)
    with tarfile.open(f"{folder}.tar.gz", 'w:gz') as archive:
        archive.add(folder, arcname=base_folder_name)
    return f"{folder}.tar.gz"

class FineTuner:
    def __init__(self, conf):
        self.conf = conf
        os.makedirs(self.conf.tmp_dir, exist_ok=True)

    def temp_file(self, name, wipe=False):
        ret = os.path.join(self.conf.tmp_dir, name)
        if wipe:
            shutil.rmtree(ret, ignore_errors=True)
        return ret

    def massage_line(self, ln, job):
        # toss our role for now, for some reason it didn't work
        # todo: check for role support in template
        j = json.loads(ln)
        
        if pr := j.get("prompt"):
            # todo: use templates properly to massage data for instruct vs chat
            j = json.loads(ln)
            cm = j["completion"]
            j = {"messages": [{"role":"user","content":pr},{"role":"assistant","content":cm}]}
        
        if "mistral" in job["model"].lower():
            j = json.loads(ln)
            j["messages"] = [m for m in j["messages"] if m["role"] != "system"]
            ln = json.dumps(j) + "\n"
        
        return ln

    def massage_fine_tune(self, file, job):
        cnt = 0
        tc = 0
        ec = 0
        training_split_pct = job.get("hyperparameters", {}).get("training_split", 0.8)

        train_file = file + ".train"
        eval_file = file + ".eval"

        with open(train_file, "w") as tf:
            with open(eval_file, "w") as ef:
                with open(file, "r") as inp:
                    ln = inp.readline(MAX_CONTEXT)
                    while ln:
                        ln = self.massage_line(ln, job)
                        cnt += 1
                        if ec and (random.random() > training_split_pct or tc <= ec):
                            tc += 1
                            tf.write(ln)
                        else:
                            ec += 1
                            ef.write(ln)
                        ln = inp.readline(MAX_CONTEXT)
        return train_file, eval_file

    async def fine_tune(self, job):
        log.info("fine tuning: %s", job)

        yield {"status": "download_data"}

        training_url = job["training_file"]
        training_file = await self.download_file(training_url)
        job["training_file"] = training_file

        q = asyncio.Queue()
       
        loop = asyncio.get_running_loop()

        log.info("spawn thread")

        t = threading.Thread(target=lambda: self._fine_tune(job, lambda res: loop.call_soon_threadsafe(q.put_nowait, res)), daemon=True)
        
        t.start()
        while True:
            res = await q.get()
            
            if res is None:
                log.info("break none")
                break
            
            yield res

        log.info("done outer loop")
        t.join()
        log.info("done await thread")

        await asyncio.sleep(2)

        shutil.rmtree(training_file, ignore_errors=True)

    def _fine_tune(self, job, cb):
        try:
            self._unsafe_fine_tune(job, cb)
        except Exception as ex:
            log.exception("error in fine tune")
            cb({"status": "error", "detail": repr(ex)})
        finally:
            log.info("cb push none")
            cb(None)
            cb(None)

    def _unsafe_fine_tune(self, job, cb):
        training_file = job["training_file"]
        train_file, eval_file = self.massage_fine_tune(training_file, job)

        base_model = job["model"]
        datasets = load_dataset("json", data_files={"train": train_file, "eval": eval_file})
        train_dataset = datasets["train"]
        eval_dataset = datasets["eval"]

        base_model_id = base_model.split(":")[0]

        # todo: use hyperparams and Q_ filter, if present, for this

        hp = job.get("hyperparameters", {})

        args = {}

        log.info("load model")
        cb({"status": "load_model"})

        args.update(dict(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ))

        bnb_config = BitsAndBytesConfig(**args)

        # todo: ideally we use llama cpp, but the cuda support for finetune isn't there

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
        )

        # sadly, does not take generators, just loads everything in ram
        tokenizer.pad_token = tokenizer.eos_token
        # todo: derive from model params
        max_length = hp.get("pad_length", 512)
        def generate_and_tokenize_prompt(prompt):
            # all input is openai formatted, and we clean it up above if needed
            pr = prompt["messages"]
            tmpl = tokenizer.apply_chat_template(pr, tokenize=False)
            result = tokenizer(
                tmpl,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            result["labels"] = result["input_ids"].copy()
            return result

        tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
        tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

        model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto", resume_download=True)

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
            r=32,
            lora_alpha=hp.get("lora_alpha", 64),
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            bias="none",
            lora_dropout=hp.get("lora_dropout", 0.05),  # Conventional
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, config)

        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        )

        accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

        model = accelerator.prepare_model(model)

        if torch.cuda.device_count() > 1:  # If more than 1 GPU
            model.is_parallelizable = True
            model.model_parallel = True

        project = "finetune"
        base_model_name = base_model_id.split("/")[-1]
        run_name = base_model_name + "-" + project + "-" + os.urandom(16).hex()
        output_dir = "./" + run_name

        tokenizer.pad_token = tokenizer.eos_token

        class EarlyStoppingCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                cb({"status": "log", "logs": logs})
                eval_loss = logs.get("eval_loss", None)
                if eval_loss is not None and eval_loss <= hp.get("stop_eval_loss", 0.05):
                    print("Early stopping criterion reached!")
                    control.should_training_stop = True

            def on_save(self, args, state, control, **kwargs):
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                log.info(f"checkpoint {checkpoint_dir}")
                cb({"status": "checkpoint"})


        trainer = transformers.Trainer(
            model=model,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            callbacks=[EarlyStoppingCallback()],
            args=transformers.TrainingArguments(
                output_dir=output_dir,
                warmup_steps=1,
                per_device_train_batch_size=hp.get("batch_size", 2),
                gradient_accumulation_steps=hp.get("accumulation_steps", 4),
                max_steps=hp.get("max_steps", -1),
                num_train_epochs=hp.get("n_epochs", 3), # use openai terminology here
                learning_rate=hp.get("learning_rate_multiplier", 2.5e-5),  # Want a small lr for finetuning
                bf16=True,
                optim="paged_adamw_8bit",
                logging_steps=25,  # When to start reporting loss
                logging_dir="./logs",  # Directory for storing logs
                save_strategy="steps",  # Save the model checkpoint every logging step
                save_steps=25,  # Save checkpoints
                save_total_limit=5,  # Save checkpoints
                load_best_model_at_end=True,
                evaluation_strategy="steps",  # Evaluate the model every logging step
                eval_steps=25,  # Evaluate and save checkpoints every 25 steps
                do_eval=True,  # Perform evaluation at the end of training
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        log.info("start train")
        cb({"status": "start_train"})
        model.config.use_cache = False  # silence the warnings
        
        trainer.train()

        tmp = self.temp_file(run_name, wipe=True)
        tokenizer.save_pretrained(tmp)


        try:
            self.return_final(run_name, model, base_model_id, cb)
        finally: 
            shutil.rmtree(output_dir, ignore_errors=True)

    def return_final(self, run_name, model, base_model_id, cb):
        log.info("return final")

        tmp = self.temp_file(run_name)
        
        # send up lora
        model.save_pretrained(tmp, safe_serialization=True)
        gz = gzip(tmp)
        with open(gz, "rb") as fil:
            while True:
                dat = fil.read(1024*64)
                if not dat:
                    break
                res = {"status": "lora", "chunk": b64enc(dat)}
                cb(res)
      
        log.info("merge weights")

        # merge weights
        
        # reload as float16 for merge
        del model
        gc.collect()

        # reload with f16
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16, local_files_only=True, device_map="auto"), tmp)
        model = model.merge_and_unload()
        
        gc.collect()
        
        shutil.rmtree(tmp)

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(tmp)
 
        model.save_pretrained(tmp)
        
        # convert to gguf for fast inference
        log.info("ggml convert")
        
        gguf_main([tmp])
        
        gg = tmp + "/ggml-model-f16.gguf"
        with open(gg, "rb") as fil:
            while True:
                dat = fil.read(1024*64)        # 16k chunks
                if not dat:
                    break
                res = {"status": "gguf", "chunk": b64enc(dat)}
                cb(res)
        
        cb({"status": "done"})
        
        shutil.rmtree(tmp, ignore_errors=True)
        
        log.info("done train")

    async def download_file(self, training_url: str) -> str:
        output_file = self.temp_file(hashlib.md5(training_url.encode()).hexdigest())
        if not os.path.exists(output_file):
            with open(output_file + ".tmp", "wb") as fh:
                async with AsyncClient() as cli:
                    res: Response = await cli.get(training_url)
                    async for chunk in res.aiter_bytes():
                        fh.write(chunk)
            os.replace(output_file + ".tmp", output_file)
        return output_file
