from typing import Any

import regex as re
from sglang import assistant, function, gen, system, user

from trl.extras.mpo import MetaRewardModel, RewardModel


class RewardModelSummarization(RewardModel):
    """
    Reward model for summarization task.
    """

    def __init__(self, reward_model_address: str, experiment_directory: str, **kwargs):
        super().__init__(
            reward_model_address=reward_model_address, experiment_directory=experiment_directory, **kwargs
        )
        prompt_path, _ = self.get_latest_rubric_path_and_iteration_index()
        self.rubric_items = self.read_rubric_items(prompt_path)

    def parse_task_descriptions_and_prompts(self, queries: list[str]) -> tuple[list[str], list[str]]:
        separation_regex = r"You are a helpful assistant\.\nuser(.+?)Bill:\n```(.+?)```"
        return self.parse_task_descriptions_and_prompts_base(queries, separation_regex)

    @function
    def rm_score(s, bill: str, response: str, rubric_items: list[str]):
        s += system("You will act as a skilled government official.")
        s += user(
            "You will assess the quality of a summary of a US Congressional and California state bill. "
            + "You will evaluate the summary by sequentially assigning a score to each rubric item. "
            + 'For each item, you need to first write a single-sentence rationale followed by a single integer score. Finish your generation with: "<EOE>".\n'
            + "Example of your output should be:\n"
            + "<reason>[Your rationale for assigned score]</reason> <score>[integer score]</score><EOE>\n"
            + f"\nBill:\n```\n{bill}\n```"
            + f"\n\nSummary of the report to be evaluated:\n```\n{response}\n```"
            + '\n\nFor each item, you need to first write a single-sentence rationale followed by a single integer score. Finish your generation with: "<EOE>".\n'
            + "Example of your output should be:\n"
            + "<reason>[Your rationale for assigned score]</reason> <score>[integer score]</score><EOE>\n"
            + "\nEnsure that you output only an integer score, enclosed within <score> and </score> tags.\n\n"
        )
        for i, item in enumerate(rubric_items):
            s += user(
                f'\nRubric Item #{i + 1}\n{item}\n\nFor this rubric item, write your evaluation feedback as one clear, concise sentence, ending with the token "<EOE>":\n'
            )
            s += assistant(gen(f"rationale_{i + 1}", temperature=0.02, max_tokens=300, stop=["<EOE>"]))
            s += user(
                "Now, using both your evaluation feedback and the rubric's scoring criteria, assign a single integer score."
            )
            s += assistant(gen(f"score_{i + 1}", temperature=0.02, regex=r"\d+"))

        evaluations = []
        for i in range(len(rubric_items)):
            try:
                eval_feedback = s[f"rationale_{i + 1}"]
                eval_score = s[f"score_{i + 1}"]
                evaluations.append(f"<reason>{eval_feedback}</reason> <score>{eval_score}</score>")
            except Exception as e:
                print(f"Error processing rubric item {i + 1}: {e}")
                evaluations.append("None")
        s.set_var("evaluations", evaluations)

    def score(
        self, queries: list[str], responses: list[str], return_evaluations: bool = True, *args, **kwargs
    ) -> tuple[list[float], list[dict[str, Any]]]:
        """
        Compute the reward score for a list of queries and responses.
        """
        assert len(queries) == len(responses)
        # queries contain the task description and writing prompt, need to separate them
        task_descriptions, bills = self.parse_task_descriptions_and_prompts(queries)
        assert len(task_descriptions) == len(bills) == len(queries)
        inputs = [
            {"bill": bill, "response": r, "rubric_items": self.rubric_items} for bill, r in zip(bills, responses)
        ]
        states = self.rm_score.run_batch(inputs, backend=self.backend)
        assert len(states) == len(inputs)

        all_evaluations = len(states) * [None]
        all_scores = len(states) * [None]
        for i, s in enumerate(states):
            try:
                evaluations = s["evaluations"]
                all_evaluations[i] = evaluations
            except Exception as e:
                print(f"Could not retrieve s['evaluations'] for state index, {i}: {e}")
                evals = [
                    "<reason> Could not retrieve evaluations from junior instructor for this sample.</reason> <score>0</score>"
                ]
                all_evaluations[i] = evals
                all_scores[i] = 0
                continue

            sub_scores = []
            for eval_index, evaluation in enumerate(evaluations):
                try:
                    m = re.search(r"<score>(.*?)<\/score>", evaluation, re.MULTILINE | re.DOTALL)
                    if m is not None:
                        score = m.group(1).strip()
                        try:
                            score = float(score)
                        except:
                            try:
                                score = float(eval(score))
                            except:
                                score = 0
                    else:
                        print(f"Could not parse <score> from: {evaluation}")
                        all_evaluations[i][eval_index] = (
                            "<reason> Could not parse score from this evaluation rubric.</reason> <score>0</score>"
                        )
                        score = 0
                except Exception as e:
                    print(f"Error processing state index, {i}: {e}")
                    print(f"state['output']: {s['output']}")
                    all_evaluations[i][eval_index] = (
                        "<reason> Could not parse score from this evaluation rubric.</reason> <score>0</score>"
                    )
                    score = 0
                sub_scores.append(score)
            all_scores[i] = sum(sub_scores)

        assert len(all_scores) == len(all_evaluations) == len(queries)
        if return_evaluations:
            return all_scores, all_evaluations
        return all_scores


class MetaRewardModelSummarization(MetaRewardModel, RewardModelSummarization):
    """
    Meta reward model for summarization task.
    """

    def __init__(self, reward_model_address: str, experiment_directory: str, **kwargs):
        super().__init__(
            reward_model_address=reward_model_address,
            experiment_directory=experiment_directory,
            **kwargs,
        )

    @function
    def mrm_prescreen(s, prescreen_prompt: str):
        s += system("You are a skilled government official.")
        s += user(prescreen_prompt)
        s += assistant(gen("verdict", choices=["good", "ok", "bad"]))

    @function
    def mrm_analyze_and_refine(s, analyze_prompt: str, refine_prompt: str):
        s += system("You are a skilled government official.")
        s += user(analyze_prompt)
        s += assistant(gen("analysis", temperature=0.02, max_tokens=2000, stop=["<EOE>", "</EOE>"]))
        s += user(refine_prompt)
        s += user(
            "First, based on the analysis, determine how many unique scoring criteria are required and provide the total as a number:"
        )
        s += assistant(gen("num_items", temperature=0.02, regex=r"\d+"))
        s += user("Now, write each criterion one by one, ensuring that no previously written criterion is repeated.")
        refined_items = []
        for i in range(1, int(s["num_items"]) + 1):
            s += user(
                f'Write the scoring criterion #{i} in fewer than 300 words, following the specified structure, and conclude with "<EOE>":'
            )
            s += assistant(gen(f"_item_{i}", temperature=0.02, max_tokens=500, stop=["<EOE>", "</EOE>"]))
            criterion = s[f"_item_{i}"]
            refined_items.append(f"<item>\n{criterion}\n</item>")
            current_context_token_length = len(s.text().split(" ")) * 1.5
            if current_context_token_length >= 31300:
                break
        refinement = "<rubric>\n" + "\n\n".join(refined_items) + "\n</rubric>"
        s.set_var("refinement", refinement)

    @function
    def mrm_merge(s, merge_prompt: str, temperature: float = 0.02):
        s += system("You are a skilled government official.")
        s += user(merge_prompt)
        s += user(
            "First, determine how many non-overlapping and distinct scoring criteria are required to comprehensively represent the given sets. Output the total number of unique criteria needed:"
        )
        s += assistant(gen("num_merged_items", temperature=temperature, regex=r"\d+"))
        s += user("Now, write each criterion one by one, ensuring that no previously written criterion is repeated.")
        merged_items = []
        for i in range(1, int(s["num_merged_items"]) + 1):
            s += user(
                f'For merged criterion #{i}, write it in fewer than 300 words, following the specified structure, and conclude with "<EOE>":'
            )
            s += assistant(gen(f"_merged_item_{i}", temperature=temperature, max_tokens=500, stop=["<EOE>", "</EOE>"]))
            criterion = s[f"_merged_item_{i}"]
            merged_items.append(f"<item>\n{criterion}\n</item>")
            current_context_token_length = len(s.text().split(" ")) * 1.5
            if current_context_token_length >= 31300:
                break
        refinement = "<rubric>\n" + "\n\n".join(merged_items) + "\n</rubric>"
        s.set_var("merged", refinement)
