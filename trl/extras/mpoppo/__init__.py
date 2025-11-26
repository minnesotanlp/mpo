import json
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from time import time
from typing import Any, Iterable, List

import jinja2
import regex as re
from natsort import natsorted
from sglang import RuntimeEndpoint, function

from .mpoppo_datasets import (
    prepare_essay_writing_dataset,
    prepare_ethical_reasoning_dataset,
    prepare_mathematical_reasoning_dataset,
    prepare_summarization_dataset,
)


def get_task_dataset(task_name: str, tokenizer, split: str):
    """
    Load the task dataset based on the task name.
    """
    assert split in ["train", "test"], f"Split '{split}' is not supported. Please choose from ['train', 'test']"
    if task_name == "essay_writing":
        dataset = prepare_essay_writing_dataset(tokenizer, split, train_size=10000)
    elif task_name == "summarization":
        dataset = prepare_summarization_dataset(tokenizer, split, train_size=10000)
    elif task_name == "math_reasoning":
        corpus_filename = "MATH_train_clustered.json" if split == "train" else "MATH_test_clustered.json"
        data_file_path = os.path.join(os.path.dirname(__file__), "corpora", "MATH", corpus_filename)
        dataset = prepare_mathematical_reasoning_dataset(tokenizer, split, data_file_path=data_file_path)
    elif task_name == "ethical_reasoning":
        corpus_filename = "train.scruples-anecdotes.jsonl" if split == "train" else "dev-test.scruples-anecdotes.jsonl"
        data_file_path = os.path.join(os.path.dirname(__file__), "corpora", "anecdotes", corpus_filename)
        dataset = prepare_ethical_reasoning_dataset(tokenizer, split, data_file_path=data_file_path, train_size=10000)
    else:
        raise ValueError(
            f"Task name '{task_name}' is not supported. Please choose from ['essay_writing', 'summarization', 'math_reasoning', 'ethical_reasoning']"
        )
    return dataset


def get_reward_model(task_name: str, reward_model_address: str | list[str], experiment_directory: str, **kwargs):
    """
    Load the reward model based on the task name.
    """
    from trl.extras.mpoppo.rm_essay_writing import RewardModelEssayWriting
    from trl.extras.mpoppo.rm_ethical_reasoning import RewardModelEthicalReasoning
    from trl.extras.mpoppo.rm_math_reasoning import RewardModelMathReasoning
    from trl.extras.mpoppo.rm_summarization import RewardModelSummarization

    if task_name == "essay_writing":
        reward_model = RewardModelEssayWriting(
            reward_model_address=reward_model_address,
            experiment_directory=experiment_directory,
        )
    elif task_name == "summarization":
        reward_model = RewardModelSummarization(
            reward_model_address=reward_model_address,
            experiment_directory=experiment_directory,
        )
    elif task_name == "math_reasoning":
        cluster_size = kwargs.get("cluster_size")
        reward_model = RewardModelMathReasoning(
            reward_model_address=reward_model_address,
            experiment_directory=experiment_directory,
            cluster_size=cluster_size,
        )
    elif task_name == "ethical_reasoning":
        reward_model = RewardModelEthicalReasoning(
            reward_model_address=reward_model_address,
            experiment_directory=experiment_directory,
        )
    else:
        raise ValueError(
            f"Task name '{task_name}' is not supported. Please choose from ['essay_writing', 'summarization', 'math_reasoning', 'ethical_reasoning']"
        )
    return reward_model


def get_meta_reward_model(task_name: str, reward_model_address: str | list[str], experiment_directory: str, **kwargs):
    """
    Load the meta reward model based on the task name.
    """
    from trl.extras.mpoppo.rm_essay_writing import MetaRewardModelEssayWriting
    from trl.extras.mpoppo.rm_ethical_reasoning import MetaRewardModelEthicalReasoning
    from trl.extras.mpoppo.rm_math_reasoning import MetaRewardModelMathReasoning
    from trl.extras.mpoppo.rm_summarization import MetaRewardModelSummarization

    if task_name == "essay_writing":
        meta_reward_model = MetaRewardModelEssayWriting(
            reward_model_address=reward_model_address,
            experiment_directory=experiment_directory,
        )
    elif task_name == "summarization":
        meta_reward_model = MetaRewardModelSummarization(
            reward_model_address=reward_model_address,
            experiment_directory=experiment_directory,
        )
    elif task_name == "math_reasoning":
        cluster_size = kwargs.get("cluster_size")
        meta_reward_model = MetaRewardModelMathReasoning(
            reward_model_address=reward_model_address,
            experiment_directory=experiment_directory,
            cluster_size=cluster_size,
        )
    elif task_name == "ethical_reasoning":
        meta_reward_model = MetaRewardModelEthicalReasoning(
            reward_model_address=reward_model_address,
            experiment_directory=experiment_directory,
        )
    else:
        raise ValueError(
            f"Task name '{task_name}' is not supported. Please choose from ['essay_writing', 'summarization', 'math_reasoning', 'ethical_reasoning']"
        )
    return meta_reward_model


class RewardModel:
    """
    Base class for reward models.
    """

    def __init__(self, reward_model_address: str | list[str], experiment_directory: str, **kwargs):
        self.reward_model_address = reward_model_address
        self.experiment_directory = experiment_directory
        self.prompts_directory = os.path.join(self.experiment_directory, "prompts")
        self.backends = self._build_backends(reward_model_address)
        self.backend = self.backends[0]

    def _parse_addresses(self, addrs: str | Iterable[str]) -> list[str]:
        if isinstance(addrs, str):
            raw = addrs
            parts = [p.strip() for p in raw.replace(";", ",").replace(" ", ",").split(",") if p.strip()]
        else:
            parts: list[str] = []
            for a in addrs:
                parts.extend([p.strip() for p in str(a).replace(";", ",").replace(" ", ",").split(",") if p.strip()])
        if not parts:
            raise ValueError("No reward model addresses provided.")
        return parts

    def _build_backends(self, addrs: str | Iterable[str]) -> list[RuntimeEndpoint]:
        parsed = self._parse_addresses(addrs)
        return [RuntimeEndpoint(addr) for addr in parsed]

    def _run_batch(self, sglang_function, inputs: List[dict[str, Any]]):
        """Fan out batch requests across available backends and preserve ordering."""
        if not inputs:
            return []
        if len(self.backends) == 1:
            return sglang_function.run_batch(inputs, backend=self.backend)

        total = len(inputs)
        num_backends = len(self.backends)
        results: list[Any] = [None] * total

        # Simple even split
        def _slice_indices(idx: int) -> range:
            start = (total * idx) // num_backends
            end = (total * (idx + 1)) // num_backends
            return range(start, end)

        with ThreadPoolExecutor(max_workers=num_backends) as pool:
            futures = []
            for b_idx, backend in enumerate(self.backends):
                idx_range = _slice_indices(b_idx)
                if len(idx_range) == 0:
                    continue
                chunk_inputs = [inputs[i] for i in idx_range]
                futures.append(
                    (
                        idx_range,
                        pool.submit(sglang_function.run_batch, chunk_inputs, backend=backend),
                    )
                )

            for idx_range, fut in futures:
                chunk_results = fut.result()
                for offset, i in enumerate(idx_range):
                    results[i] = chunk_results[offset]

        return results

    def get_latest_rubric_path_and_iteration_index(self) -> tuple[str, int]:
        """
        Get the latest evaluation rubric file path along with its iteration index.
        """
        latest_rubric_path = natsorted(glob(os.path.join(self.prompts_directory, "evaluation_rubric_iter_*.txt")))[-1]
        assert os.path.exists(latest_rubric_path), f"Latest rubric path {latest_rubric_path} does not exist!"
        iter_index_m = re.match(r"evaluation\_rubric\_iter\_(\d+)\.txt", os.path.basename(latest_rubric_path))
        assert iter_index_m is not None, f"Failed to parse iteration index from {latest_rubric_path}"
        iter_index = int(iter_index_m.group(1))
        return latest_rubric_path, iter_index

    def read_rubric_items(self, file_path: str) -> list[str]:
        with open(file_path) as f:
            rubric = f.read()
        rgx_rubric_item = r"<item>(.*?)<\/item>"
        item_matches = re.findall(rgx_rubric_item, rubric, re.MULTILINE | re.DOTALL)
        rubric_items = [m.strip() for m in item_matches]
        if len(rubric_items) < 1:
            raise ValueError(f"No rubric items parsed for {file_path}!")
        return rubric_items

    def compute_maximum_scores(self, rubric_items: list[str]) -> list[float]:
        """
        Ask the reward model to extract the highest achievable integer score for each rubric item.
        """
        from sglang import assistant, gen, system, user

        @function
        def _max_score_for_rubric_item(s, rubric_item: str):
            s += system(
                "You are a meticulous grading assistant who only extracts scoring ranges from rubric text."
            )
            s += user(
                "You will be given one rubric item that lists scoring criteria for a task.\n"
                "Read the rubric carefully and determine the maximum integer score that can be awarded for this item.\n"
                "- Look for explicit score labels such as 'Criterion for Integer Score: **6**', 'Score 5', or ranges like 0 to 5.\n"
                "- If a range is implied (e.g., labels [0,1,2,3,4,5]), return the highest endpoint.\n"
                "- Ignore examples, years, counts, or any numbers that are not a score label.\n"
                "Respond with only the integer value and nothing else.\n\n"
                "Rubric item:\n"
                f"{rubric_item}\n\n"
                "Highest integer score:"
            )
            s += assistant(gen("max_score", temperature=0.0, regex=r"\d+"))

        inputs = [{"rubric_item": item} for item in rubric_items]
        states = self._run_batch(_max_score_for_rubric_item, inputs)

        max_scores: list[float] = []
        for idx, state in enumerate(states):
            try:
                raw_score = state["max_score"]
            except Exception as e:
                print(f"Could not retrieve max_score for rubric item #{idx + 1}: {e}")
                max_scores.append(0.0)
                continue

            try:
                score_val = float(raw_score)
            except Exception:
                m = re.search(r"\d+(?:\.\d+)?", str(raw_score))
                if m:
                    score_val = float(m.group(0))
                else:
                    print(f"Could not parse numeric max_score for rubric item #{idx + 1}: {raw_score}")
                    score_val = 0.0
            max_scores.append(score_val)

        if len(max_scores) != len(rubric_items):
            print(
                "Warning: Number of extracted max scores does not match rubric items; padding/truncating to match length."
            )
            max_scores = max_scores[: len(rubric_items)] + [0.0] * (
                len(rubric_items) - len(max_scores)
            )

        return max_scores

    def parse_task_descriptions_and_prompts_base(
        self, queries: list[str], separation_regex: str
    ) -> tuple[list[str], list[str]]:
        task_descriptions = []
        writing_prompts = []
        for q in queries:
            m = re.search(separation_regex, q, re.MULTILINE | re.DOTALL)
            if m is not None:
                task_description = m.group(1).strip()
                writing_prompt = m.group(2).strip()
                task_descriptions.append(task_description)
                writing_prompts.append(writing_prompt)
            else:
                raise ValueError(f"Could not parse task description and writing prompt from: {q}")
        return task_descriptions, writing_prompts

    def parse_task_descriptions_and_prompts(self, queries: list[str]) -> tuple[list[str], list[str]]:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the parse_task_descriptions_and_prompts() method."
        )

    @function
    def rm_score(s, task_description: str, writing_prompt: str, response: str, rubric_items: list[str]):
        raise NotImplementedError("rm_score() method must be implemented.")

    def score(
        self, queries: list[str], responses: list[str], return_evaluations: bool = True, **kwargs
    ) -> list[float]:
        """
        Compute the reward score for a list of queries and responses.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the score() method.")


class MetaRewardModel(RewardModel):
    """
    Base class for meta reward models.
    """

    def __init__(self, reward_model_address: str, experiment_directory: str, **kwargs):
        super().__init__(
            reward_model_address=reward_model_address,
            experiment_directory=experiment_directory,
            **kwargs,
        )
        self.rollouts_directory = os.path.join(self.experiment_directory, "rollouts")
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(self.prompts_directory))
        mrm_prescreen_filename = "mrm_prescreen.txt"
        mrm_analyze_filename = "mrm_analyze.txt"
        mrm_refine_filename = "mrm_refine.txt"
        mrm_merge_filename = "mrm_merge.txt"
        try:
            self.prescreen_template = env.get_template(mrm_prescreen_filename)
        except jinja2.TemplateNotFound as e:
            print(f"Template {e}, Skipped.")
        try:
            self.analyze_template = env.get_template(mrm_analyze_filename)
        except jinja2.TemplateNotFound as e:
            print(f"Template {e}, Skipped.")
        try:
            self.refine_template = env.get_template(mrm_refine_filename)
        except jinja2.TemplateNotFound as e:
            print(f"Template {e}, Skipped.")
        try:
            self.merge_template = env.get_template(mrm_merge_filename)
        except jinja2.TemplateNotFound as e:
            print(f"Template {e}, Skipped.")

    @function
    def mrm_prescreen(s, prescreen_prompt: str):
        raise NotImplementedError("mrm_prescreen() method must be implemented.")

    @function
    def mrm_analyze_and_refine(s, analyze_prompt: str, refine_prompt: str):
        raise NotImplementedError("mrm_analyze_and_refine() method must be implemented.")

    @function
    def mrm_merge(s, merge_prompt: str, temperature: float = 0.02):
        raise NotImplementedError("mrm_merge() method must be implemented.")

    def meta_evaluate_and_update(
        self,
        batch_index: int,
        return_evaluations: bool = True,
        num_samples: int = 20,
        do_prescreening: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Meta-evaluate the model, update the evaluation rubric, and return the analyses.
        """
        current_junior_prompt_path, current_iter_no = self.get_latest_rubric_path_and_iteration_index()
        with open(current_junior_prompt_path) as f:
            current_junior_prompt = f.read().strip()

        rollouts_paths = glob(os.path.join(self.rollouts_directory, f"rollouts-{batch_index}-*.json"))

        inputs = []
        prescreen_inputs = []
        for path in rollouts_paths:
            with open(path) as f:
                rollouts = json.load(f)
            queries = [sample["queries"] for sample in rollouts]
            _task_descriptions, _writing_prompts = self.parse_task_descriptions_and_prompts(queries)

            for i, sample in enumerate(rollouts):
                _input = {
                    "task_description": _task_descriptions[i],
                    "student_prompt": _writing_prompts[i],
                    "student_generation": sample["student_responses"],
                    "junior_prompt": current_junior_prompt,
                    "junior_score": sample["junior_scores"],
                }
                inputs.append(_input)
                prescreen_inputs.append(self.prescreen_template.render(_input))

        if do_prescreening:
            ### Prescreening
            print(f"Prescreening {len(prescreen_inputs)} samples...")
            start_time = time()
            states = self._run_batch(self.mrm_prescreen, [{"prescreen_prompt": _input} for _input in prescreen_inputs])
            assert len(states) == len(inputs) == len(prescreen_inputs)
            prescreened_verdicts = []
            prescreened_ok_indices = []
            prescreened_bad_indices = []
            for i, s in enumerate(states):
                try:
                    s["verdict"] = s["verdict"].strip()
                except Exception as e:
                    print(f"Could not retrieve s['verdict'] for state index: {e}")
                    s.set_var("verdict", "None")
                prescreened_verdicts.append(s["verdict"])
                if s["verdict"] == "ok":
                    prescreened_ok_indices.append(i)
                elif s["verdict"] == "bad":
                    prescreened_bad_indices.append(i)
            prescreened_verdicts_counter = Counter(prescreened_verdicts)

            end_time = time()
            prescreen_time = (end_time - start_time) / 60
            print(f"Prescreening took {prescreen_time:.2f} minutes for {len(prescreen_inputs)} samples.")
            print(f"Prescreened verdicts dist.: {prescreened_verdicts_counter}")

            random.shuffle(prescreened_bad_indices)
            random.shuffle(prescreened_ok_indices)
            selected_indices = prescreened_bad_indices[:num_samples]
            if len(selected_indices) < num_samples:
                selected_indices += prescreened_ok_indices[: num_samples - len(selected_indices)]
        else:
            selected_indices = random.sample(range(len(inputs)), num_samples)

        ### Analysis and Refinement
        print("Analysis and Refinment step...")
        start_time = time()
        analyze_prompt_all = []
        refine_prompt_all = []
        for index in selected_indices:
            sample = {"max_words": 1000, **inputs[index]}
            analyze_prompt = self.analyze_template.render(sample)
            analyze_prompt_all.append(analyze_prompt)

            refine_prompt = self.refine_template.render({})
            refine_prompt_all.append(refine_prompt)

        analysis_and_refinement_inputs = [
            {"analyze_prompt": a_p, "refine_prompt": r_p} for a_p, r_p in zip(analyze_prompt_all, refine_prompt_all)
        ]
        states = self._run_batch(self.mrm_analyze_and_refine, analysis_and_refinement_inputs)
        analyses = []
        refinements = []
        for i, s in enumerate(states):
            analysis = s["analysis"]
            analyses.append(analysis)
            refinement = s["refinement"]
            refinements.append(f"Junior Instructor's Scoring Criteria Set #{i + 1}:\n{refinement}\n")
        assert len(analyses) == len(refinements) == len(selected_indices)
        end_time = time()
        analysis_time = (end_time - start_time) / 60
        print(f"Analysis and Refinement took {analysis_time:.2f} minutes for {len(selected_indices)} samples.")

        ### Merge the refinements
        print("Merge step...")
        start_time = time()
        ### Need to reserve some space in the input context for merged prompt
        joined_prompts = self.join_under_word_limit(refinements, int(25000 / 1.5), sep="\n===\n")
        merge_prompt = self.merge_template.render({"multiple_sets": joined_prompts + "\n```"})
        state = self.mrm_merge.run(merge_prompt=merge_prompt, backend=self.backend)
        merged_criteria = state["merged"]

        # Keep only unique criteria
        regex = r"<item>(.*?)<\/item>"
        matches = re.findall(regex, merged_criteria, re.MULTILINE | re.DOTALL)
        rubric_items = {m.strip() for m in matches}
        rubric_items = [f"<item>\n{item}\n</item>" for item in rubric_items]
        merged_criteria = "<rubric>\n" + "\n\n".join(rubric_items) + "</rubric>"

        # Save newer version
        next_junior_prompt_path = os.path.join(
            self.prompts_directory, f"evaluation_rubric_iter_{current_iter_no + 1}.txt"
        )
        with open(next_junior_prompt_path, "w") as f:
            f.write(f"{merged_criteria.strip()}\n")
        end_time = time()
        merge_time = (end_time - start_time) / 60
        print(f"Merge took {merge_time:.2f} minutes. New version saved to {next_junior_prompt_path}")

        if return_evaluations:
            assert len(analyses) == len(refinements) == len(selected_indices)
            logs = []
            for i, selected_index in enumerate(selected_indices):
                log = {
                    "junior_prompt": current_junior_prompt,
                    "meta_analysis": analyses[i],
                    "meta_refinement": refinements[i],
                    **inputs[selected_index],
                }
                if do_prescreening:
                    log["prescreened_verdict"] = prescreened_verdicts[selected_index]
                logs.append(log)
            return logs

    def join_under_word_limit(self, refinements: list[str], N: int, sep: str = "\n===\n") -> str:
        """
        Join strings from `refinements` with `sep`, stopping before the total
        whitespace‑separated word count exceeds `N`.

        Parameters
        ----------
        refinements : list[str]
            The strings to join.
        N : int
            Maximum total number of words allowed in the joined result.
        sep : str, optional
            Separator to place between pieces. Defaults to "\n===\n".

        Returns
        -------
        str
            The joined string whose word count is ≤ N.
        """
        joined_parts = []
        word_count = 0

        for chunk in refinements:
            words_in_chunk = len(chunk.split())
            if word_count + words_in_chunk > N:
                break
            joined_parts.append(chunk)
            word_count += words_in_chunk

        return sep.join(joined_parts)
