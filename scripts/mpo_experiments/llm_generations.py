import gc
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator, PartialState
from accelerate.utils import gather_object
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, HfArgumentParser, TrainingArguments

from trl.extras.mpo import get_task_dataset
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE, MPODataCollatorWithPadding, generate


"""Example
accelerate launch --config_file examples/accelerate_configs/multi_gpu_4gpus.yaml scripts/mpo_experiments/llm_generations.py \
	--task_name "essay_writing" \
	--exp_name "expert-32b" \
	--dataset_split "test" \
	--response_length 400 \
	--batch_size 64 \
	--tokenizer_name "Qwen/Qwen2-1.5B-Instruct" \
	--model_path "/home/elicer/Development/trl/models/essay_writing/ppo/expert-32b/checkpoints/407" \
	--output_dir "/home/elicer/Development/trl/results/generations/essay_writing"
"""


def seed_everything(seed: int = 42):
    """
    Set the random seed for reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class InferenceConfig(TrainingArguments):
    r"""
    Config class for inferencing trained models.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        task_name (`str`, *optional*, defaults to `None`):
            Name of the task. Should be one of ['essay_writing', 'summarization', 'math_reasoning', 'ethical_reasoning']
        exp_name (`str`, *optional*, defaults to `None`):
            Name of the experiment.
        dataset_split (`str`, *optional*, defaults to `None`):
            Split of the dataset. Should be one of ['train', 'test']
        response_length (`int`, *optional*, defaults to `400`):
            Length of the response.
        batch_size (`int`, *optional*, defaults to `128`):
            Overall batch size.
        tokenizer_name (`str`, *optional*, defaults to `Qwen/Qwen2-1.5B-Instruct`):
            Name of the tokenizer.
        model_path (`str`, *optional*, defaults to `None`):
            Path to the model.
        output_dir (`str`, *optional*, defaults to `None`):
            The output directory where the model generations will be written.
    """

    task_name: str = field(
        default=None,
        metadata={
            "help": "Name of the task. Should be one of ['essay_writing', 'summarization', 'math_reasoning', 'ethical_reasoning']"
        },
    )
    exp_name: str = field(
        default=None,
        metadata={"help": "Name of the experiment."},
    )
    dataset_split: str = field(
        default=None,
        metadata={"help": "Split of the dataset. Should be one of ['train', 'test']"},
    )
    response_length: int = field(
        default=400,
        metadata={"help": "Length of the response."},
    )
    batch_size: int = field(
        default=128,
        metadata={"help": "Overall batch size."},
    )
    tokenizer_name: str = field(
        default="Qwen/Qwen2-1.5B-Instruct",
        metadata={"help": "Name of the tokenizer."},
    )
    model_path: str = field(
        default=None,
        metadata={"help": "Path to the model."},
    )
    output_dir: str = field(
        default=None,
        metadata={"help": "The output directory where the model generations will be written."},
    )


if __name__ == "__main__":
    seed_everything(42)
    parser = HfArgumentParser(InferenceConfig)
    args = parser.parse_args_into_dataclasses()[0]
    task_name = args.task_name
    split = args.dataset_split
    exp_name = args.exp_name
    output_dir = args.output_dir

    if "Qwen/" not in args.model_path:
        assert os.path.exists(args.model_path)
    with PartialState().local_main_process_first():
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=False)

    assert task_name in [
        "essay_writing",
        "summarization",
        "math_reasoning",
        "ethical_reasoning",
    ], (
        f"Task name '{task_name}' is not supported. Please choose from ['essay_writing', 'summarization', 'math_reasoning', 'ethical_reasoning']"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, padding_side="left", trust_remote_code=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    with PartialState().local_main_process_first():
        dataset = get_task_dataset(task_name, tokenizer, split)

    accelerator = Accelerator()

    gen_save_path = os.path.join(output_dir, f"{exp_name}.{split}.generations.jsonl")
    assert not os.path.exists(gen_save_path)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        # device_map="auto",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=MPODataCollatorWithPadding(tokenizer),
        drop_last=False,
    )

    dataloader, model = accelerator.prepare(dataloader, model)

    generation_config = GenerationConfig(
        max_new_tokens=args.response_length,
        temperature=(0.01 + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    table = defaultdict(list)
    for batch in tqdm(dataloader, total=len(dataloader)):
        query = batch["input_ids"]
        if "gold_answers" in batch:
            gold_answers = batch["gold_answers"]
        if "clean_summary" in batch:
            clean_summary = batch["clean_summary"]
        if "verdict" in batch:
            verdict = batch["verdict"]
        with torch.no_grad():
            context_length = query.shape[1]
            if "Qwen/" in args.model_path:
                model = accelerator.unwrap_model(model)
            query_response, _ = generate(
                model,
                query,
                tokenizer.pad_token_id,
                generation_config,
            )
            response = query_response[:, context_length:]
            postprocessed_response = response

            table["query"].extend(gather_object(tokenizer.batch_decode(query, skip_special_tokens=True)))
            table["model_response"].extend(
                gather_object(tokenizer.batch_decode(postprocessed_response, skip_special_tokens=True))
            )
            if "gold_answers" in batch:
                table["gold_answer"].extend(
                    gather_object(tokenizer.batch_decode(gold_answers, skip_special_tokens=True))
                )
            if "clean_summary" in batch:
                table["clean_summary"].extend(
                    gather_object(tokenizer.batch_decode(clean_summary, skip_special_tokens=True))
                )
            if "verdict" in batch:
                table["verdict"].extend(gather_object(verdict.detach().cpu().tolist()))

    df = pd.DataFrame(table)
    with open(gen_save_path, "w") as f:
        f.write(df.to_json(orient="records", lines=True))

    del model
    del dataloader
    del table
    del df
    gc.collect()
    torch.cuda.empty_cache()
