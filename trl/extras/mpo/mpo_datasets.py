import multiprocessing as mp

from datasets import load_dataset


def prepare_essay_writing_dataset(tokenizer, split: str = "test", train_size: int = None):
    _dataset = load_dataset("zaemyung/writing_prompts_collection")[split]

    def _prepare_dataset(dataset, tokenizer):
        def tokenize(sample):
            input_ids = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": "You will act as an English writer and compose either an essay or a story depending on the instruction given below. Your essay should be no more than 350 words.\n\nInstructions:\n"
                        + sample["prompt"]
                        + "\n\nYour Writing:\n",
                    }
                ],
                padding=False,
                add_generation_prompt=True,
            )
            return {"input_ids": input_ids, "lengths": len(input_ids)}

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            num_proc=mp.cpu_count(),
            load_from_cache_file=True,
        )

    dataset = _prepare_dataset(_dataset, tokenizer)
    if split == "train":
        dataset = dataset.shuffle(seed=42)
    assert dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"
    if split == "train" and train_size is not None:
        return dataset.select(range(train_size))

    return dataset


def prepare_mathematical_reasoning_dataset(tokenizer, split: str, data_file_path: list[str]):
    def extract_boxed_content(string: str):
        start_token = r"\boxed{"
        start = string.find(start_token)
        if start == -1:
            return None

        # Start after "\boxed{"
        i = start + len(start_token)
        brace_count = 1
        content = []

        while i < len(string):
            char = string[i]
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    break
            content.append(char)
            i += 1

        return "".join(content)

    def tokenize(sample):
        answer = extract_boxed_content(sample["solution"])
        if answer is None:
            print(sample)
        input_ids = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": 'You will take on the role of a mathematician, solving a given math problem with clear mathematical reasoning, and present the final answer enclosed in "\\boxed{ }". Your solution and answer should be in Latex equation format. For example, followed by a solution, an answer can be presented as : "\\boxed{3 \\text{cm}}" or "\\boxed{\\frac{2}{3}}".\n\nProblem:\n'
                    + sample["problem"]
                    + '\n\nYour solution with answer (enclosed in, literally, "\\boxed{ }"):\n',
                }
            ],
            padding=False,
            add_generation_prompt=True,
        )
        return {
            "input_ids": input_ids,
            "lengths": len(input_ids),
            "answer": tokenizer.encode(answer.strip()),
            "domain_ids": sample["domain_id"],
            "cluster_ids": sample["cluster_id"],
            "solution": tokenizer.encode(sample["solution"], add_special_tokens=False, padding=False),
        }

    dataset = load_dataset("json", data_files=data_file_path, split="train")
    dataset = dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        num_proc=mp.cpu_count(),
        load_from_cache_file=True,
    )
    if split == "train":
        dataset = dataset.shuffle(seed=42)
    return dataset


def prepare_summarization_dataset(tokenizer, split: str = "test", train_size: int = None):
    def tokenize(sample):
        input_ids = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "Acting as an experienced government official, write a concise and faithful summary of the following US Congressional and California state bill. "
                    + "Your summary should be less than 400 words.\n\nBill:\n```\n"
                    + sample["clean_text"]
                    + "\n```"
                    + "\n\nYour summary should be less than 400 words."
                    + "\nYour Summary:\n",
                }
            ],
            padding=False,
            add_generation_prompt=True,
        )
        return {
            "input_ids": input_ids,
            "lengths": len(input_ids),
            "clean_summary": tokenizer.encode(sample["clean_summary"], add_special_tokens=False, padding=False),
        }

    raw_dataset = load_dataset("duccd/billsum-clean")
    dataset = raw_dataset[split]
    tokenized_dataset = dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        num_proc=mp.cpu_count(),
        load_from_cache_file=True,
    )
    tokenized_dataset = tokenized_dataset.filter(
        lambda sample: len(sample["input_ids"]) <= 4000,
        load_from_cache_file=True,
        num_proc=mp.cpu_count(),
    )
    if split == "train":
        tokenized_dataset = tokenized_dataset.shuffle(seed=42)
    if split == "train" and train_size is not None:
        return tokenized_dataset.select(range(train_size))
    return tokenized_dataset


def prepare_ethical_reasoning_dataset(tokenizer, split: str, data_file_path: str, train_size: int = None):
    def _filter_sample(sample):
        try:
            assert sample["binarized_label"] in ["RIGHT", "WRONG"]
            assert sample["title"] is not None
            assert sample["text"] is not None
            assert sample["action"]["description"] is not None
        except:
            return False
        return True

    def tokenize(sample):
        verdict = 1 if sample["binarized_label"] == "RIGHT" else 0
        sample_dict = {
            "role": "user",
            "content": (
                "Assume the role of a moral philosopher tasked with evaluating the ethical nature of an action described in the anecdote below. "
                'Your response must include a concise ethical analysis, followed by a clear moral verdict (as "RIGHT" or "WRONG"), depending on whether the action in question is ethically right or wrong.\n\n'
                "Followed by the reasoning, you need to enclose your final verdict within <verdict> </verdict> tag. "
                'An output example would be: "(Your reasoning goes here.)<verdict>WRONG</verdict>\n"'
                "The reasoning should be no longer than 300 words.\n\n"
                "### Anecdote\n"
                f"Title: {sample['title']}\n"
                f"{sample['text']}\n\n"
                "### Action to Evaluate\n"
                f'"{sample["action"]["description"]}"\n\n'
                '### Ethical Reasoning and Verdict (Write ethical reasoning first and then give the verdict as "<verdict>RIGHT</verdict>" or "<verdict>WRONG</verdict>")\n'
            ),
        }
        input_ids = tokenizer.apply_chat_template(
            [sample_dict],
            padding=False,
            add_generation_prompt=True,
        )
        return {"input_ids": input_ids, "lengths": len(input_ids), "verdict": verdict}

    dataset = load_dataset("json", data_files=data_file_path, split="train")
    dataset = dataset.filter(
        _filter_sample,
        load_from_cache_file=True,
        num_proc=mp.cpu_count(),
    )
    tokenized_dataset = dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        num_proc=mp.cpu_count(),
        load_from_cache_file=True,
    )
    if split == "train":
        tokenized_dataset = tokenized_dataset.shuffle(seed=42)
    if split == "train" and train_size is not None:
        return tokenized_dataset.select(range(train_size))
    return tokenized_dataset
