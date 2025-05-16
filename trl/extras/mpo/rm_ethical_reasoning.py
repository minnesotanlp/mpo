from typing import Any

import regex as re
from sglang import assistant, function, gen, system, user

from trl.extras.mpo import MetaRewardModel, RewardModel


class RewardModelEthicalReasoning(RewardModel):
    """
    Reward model for essay writing task.
    """

    def __init__(self, reward_model_address: str, experiment_directory: str, **kwargs):
        super().__init__(
            reward_model_address=reward_model_address, experiment_directory=experiment_directory, **kwargs
        )
        prompt_path, _ = self.get_latest_rubric_path_and_iteration_index()
        self.rubric_items = self.read_rubric_items(prompt_path)

    def parse_task_descriptions_and_prompts(self, queries: list[str]) -> tuple[list[str], list[str]]:
        separation_regex = (
            r"You are a helpful assistant.\nuser(.+?)### Anecdote\n(.+?)### Ethical Reasoning and Verdict"
        )
        return self.parse_task_descriptions_and_prompts_base(queries, separation_regex)

    @function
    def rm_score(s, anec: str, response: str, rubric_items: list[str]):
        s += system("You are an insightful moral philosopher.")
        s += user(
            "You are tasked with assessing the quality of a moral reasoning and its corresponding verdict that evaluate the ethical nature of an action described in the anecdote below.\n"
            + "Evaluate the response by scoring each rubric item individually.\n\n"
            + "For each rubric item:\n"
            + "- Write a single-sentence rationale explaining your score. This sentence should be enclosed in <reason> and </reason> tags.\n"
            + "- Follow this with an integer score, enclosed in <score> and </score> tags.\n"
            + "- End each item with the marker: <EOE>\n\n"
            + "### Format Example:\n"
            "<reason>[Your rationale for assigned score]</reason> <score>[integer score]</score><EOE>\n\n"
            + f"### Anecdote:\n{anec}\n\n"
            + f"### Moral Reasoning and Verdict:\n{response}\n\n"
            + "Important: Output only the rationale, the score (within tags), and the <EOE> marker for each rubric item. No additional commentary or formatting outside the specified structure."
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
        task_descriptions, anecdotes = self.parse_task_descriptions_and_prompts(queries)
        assert len(task_descriptions) == len(anecdotes) == len(queries)
        rgx_verdict = r"(<verdict>(RIGHT|WRONG)</verdict>)"
        rgx_anec = r"### Anecdote(.+)### Ethical Analysis and Verdict"

        indices_for_llm = []
        inputs_for_llm = []
        all_scores = []
        all_evaluations = []
        for i, (parsed_anec, r) in enumerate(zip(anecdotes, responses)):
            verdict_part = re.search(rgx_verdict, r, re.MULTILINE | re.DOTALL)
            if verdict_part is None:
                evals = ["<reason> Reasoning is missing verdict.</reason> <score>-10</score>"]
                all_evaluations.append(evals)
                all_scores.append(-10)
                continue

            response_without_verdict = re.sub(rgx_verdict, "", r).strip()
            if len(response_without_verdict.split()) < 30:
                evals = ["<reason> Reasoning is too short.</reason> <score>-10</score>"]
                all_evaluations.append(evals)
                all_scores.append(-10)
                continue

            # otherwise, we need to run the LLM to get the score
            indices_for_llm.append(i)
            try:
                anec = re.search(rgx_anec, parsed_anec, re.MULTILINE | re.DOTALL).group(1)
            except:
                anec = parsed_anec
            inputs_for_llm.append({"anec": anec, "response": r, "rubric_items": self.rubric_items})
            all_evaluations.append("dummy state")
            all_scores.append("dummy score")
        assert len(queries) == len(all_evaluations) == len(all_scores)
        assert len(inputs_for_llm) == len(indices_for_llm)

        states_from_llm = self.rm_score.run_batch(inputs_for_llm, backend=self.backend)
        assert len(states_from_llm) == len(inputs_for_llm) == len(indices_for_llm)
        for all_states_index, s in zip(indices_for_llm, states_from_llm):
            assert all_evaluations[all_states_index] == "dummy state"
            assert all_scores[all_states_index] == "dummy score"
            try:
                evaluations = s["evaluations"]
                all_evaluations[all_states_index] = evaluations
            except Exception as e:
                print(f"Could not retrieve s['evaluations'] for state index, {i}: {e}")
                evals = [
                    "<reason> Could not retrieve evaluations from junior instructor for this sample.</reason> <score>0</score>"
                ]
                all_evaluations[all_states_index] = evals
                all_scores[all_states_index] = 0
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
                        all_evaluations[all_states_index][eval_index] = (
                            "<reason> Could not parse score from this evaluation rubric.</reason> <score>0</score>"
                        )
                        score = 0
                except Exception as e:
                    print(f"Error processing state index, {i}: {e}")
                    print(f"state['output']: {s['output']}")
                    all_evaluations[all_states_index][eval_index] = (
                        "<reason> Could not parse score from this evaluation rubric.</reason> <score>0</score>"
                    )
                    score = 0
                sub_scores.append(score)
            all_scores[all_states_index] = sum(sub_scores)

        assert len(all_scores) == len(all_evaluations) == len(queries)
        if return_evaluations:
            return all_scores, all_evaluations
        return all_scores


class MetaRewardModelEthicalReasoning(MetaRewardModel, RewardModelEthicalReasoning):
    """
    Meta reward model for essay writing task.
    """

    def __init__(self, reward_model_address: str, experiment_directory: str, **kwargs):
        super().__init__(
            reward_model_address=reward_model_address,
            experiment_directory=experiment_directory,
            **kwargs,
        )

    @function
    def mrm_prescreen(s, prescreen_prompt: str):
        s += system("You are an insightful moral philosopher.")
        s += user(prescreen_prompt)
        s += assistant(gen("verdict", choices=["good", "ok", "bad"]))

    @function
    def mrm_analyze_and_refine(s, analyze_prompt: str, refine_prompt: str):
        s += system("You are an insightful moral philosopher.")
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
        s += system("You are an insightful moral philosopher.")
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
