import base64
import hashlib
import json
import logging
import os
import random

import transformers
from datasets import load_dataset
from httpx import AsyncClient, Response

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from ai_worker.jsonlines import load_jsonlines

MAX_CONTEXT = 300000

log = logging.getLogger(__name__)


class FineTuner:
    def __init__(self, conf):
        self.conf = conf
        os.makedirs(self.conf.tmp_dir, exist_ok=True)

    def temp_file(self, name):
        return os.path.join(self.conf.tmp_dir, name)

    def massage_line(self, ln, job):
        return ln

    def massage_fine_tune(self, file, job):
        cnt = 0
        training_split_pct = job.get("hyperparameters", {}).get("training_split", 0.8)

        train_file = file + ".train"
        eval_file = file + ".eval"

        with open(train_file, "w") as tf:
            with open(eval_file, "w") as ef:
                with open(file, "r") as inp:
                    ln = inp.readline(MAX_CONTEXT)
                    ln = self.massage_line(ln, job)
                    while ln:
                        cnt += 1
                        if random.random() > training_split_pct:
                            tf.write(ln)
                        else:
                            ef.write(ln)
                        ln = inp.readline(MAX_CONTEXT)
                        ln = self.massage_line(ln, job)
        return train_file, eval_file

    async def fine_tune(self, job):
        log.info("fine tuning: %s", job)

        yield {"status": "downloading_data"}

        base_model = job["model"]
        training_url = job["training_file"]
        training_file = await self.download_file(training_url)

        train_file, eval_file = self.massage_fine_tune(training_file, job)

        train_dataset = load_jsonlines(open(train_file))
        eval_dataset = load_jsonlines(open(eval_file))

        # todo: use user's model request

        base_model_id = "mistralai/Mistral-7B-v0.1"

        # todo: use hyperparams and Q_ filter, if present, for this

        hp = job.get("hyperparameters", {})

        args = {}

        yield {"status": "loading_model"}

        args.update(dict(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ))

        bnb_config = BitsAndBytesConfig(**args)

        # todo: ideally we use llama cpp, but the cuda support for finetune isn't there

        model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
        )

        train_dataset = tokenizer.apply_chat_template(train_dataset)
        eval_dataset = tokenizer.apply_chat_template(eval_dataset)

        tokenizer.pad_token = tokenizer.eos_token

        max_length = 512

        def generate_and_tokenize_prompt(prompt):
            result = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            result["labels"] = result["input_ids"].copy()
            return result

        tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
        tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

        from peft import prepare_model_for_kbit_training

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=32,
            lora_alpha=64,
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
            lora_dropout=0.05,  # Conventional
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, config)

        from accelerate import FullyShardedDataParallelPlugin, Accelerator
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        )

        accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

        model = accelerator.prepare_model(model)

        if torch.cuda.device_count() > 1:  # If more than 1 GPU
            model.is_parallelizable = True
            model.model_parallel = True

        project = "journal-finetune"
        base_model_name = "mistral"
        run_name = base_model_name + "-" + project
        output_dir = "./" + run_name

        tokenizer.pad_token = tokenizer.eos_token

        trainer = transformers.Trainer(
            model=model,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            args=transformers.TrainingArguments(
                output_dir=output_dir,
                warmup_steps=1,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=1,
                max_steps=500,
                learning_rate=2.5e-5,  # Want a small lr for finetuning
                bf16=True,
                optim="paged_adamw_8bit",
                logging_steps=25,  # When to start reporting loss
                logging_dir="./logs",  # Directory for storing logs
                save_strategy="steps",  # Save the model checkpoint every logging step
                save_steps=25,  # Save checkpoints
                evaluation_strategy="steps",  # Evaluate the model every logging step
                eval_steps=25,  # Evaluate and save checkpoints every 50 steps
                do_eval=True,  # Perform evaluation at the end of training
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train()

        res = {"status": "done", "checkpoint": str(base64.b64encode(b"checkpoint"))}
        yield res

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
