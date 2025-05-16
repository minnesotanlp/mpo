import json
import os
import random
from glob import glob
from time import time
from typing import Any

import jinja2
import numpy as np
import regex as re
from natsort import natsorted
from sglang import assistant, function, gen, system, user

from trl.extras.mpo import MetaRewardModel, RewardModel


math_domains_to_idx = {
    "algebra": 0,
    "counting_and_probability": 1,
    "geometry": 2,
    "intermediate_algebra": 3,
    "number_theory": 4,
    "prealgebra": 5,
    "precalculus": 6,
}
idx_to_math_domains = {v: k for k, v in math_domains_to_idx.items()}


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2


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


class RewardModelMathReasoning(RewardModel):
    """
    Reward model for mathematical reasoning task.
    """

    def __init__(self, reward_model_address: str, experiment_directory: str, **kwargs):
        super().__init__(
            reward_model_address=reward_model_address, experiment_directory=experiment_directory, **kwargs
        )
        self.cluster_size = kwargs.get("cluster_size", None)
        self.domain_cluster_guidelines, _ = self.read_meta_level_guidelines_and_iter_nos()
        junior_env = jinja2.Environment(loader=jinja2.FileSystemLoader(self.prompts_directory))
        self.score_plan_template = junior_env.get_template("rm_plan.txt")
        self.score_execute_template = junior_env.get_template("rm_execute.txt")

    def read_meta_level_guidelines_and_iter_nos(self) -> tuple[dict[str, dict[int, str]], dict[str, dict[int, int]]]:
        """
        Retrieve the latest metalevel guidelines for every cluster per math domain.
        """

        def _read_latest_guideline_and_iter_no(domain_cluster_dir):
            latest_guideline_path = natsorted(glob(os.path.join(domain_cluster_dir, "evaluation_rubric_iter_*")))[-1]
            with open(latest_guideline_path) as inf:
                guideline_prompt = inf.read()
            iter_no_m = re.match(r"evaluation\_rubric\_iter\_(\d+)\.txt", os.path.basename(latest_guideline_path))
            if iter_no_m is not None:
                iter_no = int(iter_no_m.group(1))
            else:
                iter_no = None
            return guideline_prompt, iter_no

        domain_cluster_guidelines = {math_domain: {} for math_domain in math_domains_to_idx.keys()}
        domain_cluster_iter_nos = {math_domain: {} for math_domain in math_domains_to_idx.keys()}
        for math_domain in domain_cluster_guidelines.keys():
            for cluster_idx in range(1, self.cluster_size + 1):
                domain_cluster_dir = os.path.join(self.prompts_directory, math_domain, f"cluster-{cluster_idx}")
                guidelines, current_iter_no = _read_latest_guideline_and_iter_no(domain_cluster_dir)
                domain_cluster_guidelines[math_domain][cluster_idx] = guidelines
                domain_cluster_iter_nos[math_domain][cluster_idx] = current_iter_no
        return domain_cluster_guidelines, domain_cluster_iter_nos

    def parse_task_descriptions_and_prompts(self, queries: list[str]) -> tuple[list[str], list[str]]:
        separation_regex = r"user(.+?)Problem:(.+?)Your solution with answer"
        return self.parse_task_descriptions_and_prompts_base(queries, separation_regex)

    @function
    def rm_score(s, plan_prompt: str, execute_prompt: str):
        s += user("You are a skilled mathematician.")
        s += user(plan_prompt)
        s += assistant(gen("plan", temperature=0.02, max_tokens=2000, stop=["<EOE>"]))
        plan = f"[Start of Evaluation Plan]\n{s['plan'].strip()}\n[End of Evaluation Plan]\n"
        s.set_var("plan", plan)
        s += user(f"The tailored [Evaluation Plan] is the following:\n{plan}\n")
        s += user(execute_prompt)
        s += assistant(gen("execute", temperature=0.02, max_tokens=2000, stop=["<EOE>"]))
        execute = f"[Start of Evaluation Execution]\n{s['execute'].strip()}\n[End of Evaluation Execution]\n"
        s.set_var("execute", execute)
        s += user(f"You executed the plan with the following steps:\n{execute}\n")
        s += user(
            "Finally, based on your executed results, assign a final evaluation score as a single integer from 0 to 5, where 0 indicates the lowest quality and 5 the highest. Output only the integer.\nYour Score: "
        )
        s += assistant(gen("score", temperature=0.02, regex=r"[0-5]"))

    def score(
        self,
        queries: list[str],
        responses: list[str],
        return_evaluations: bool = True,
        gold_answers: list[str] = None,
        solutions: list[str] = None,
        domain_ids: list[int] = None,
        cluster_ids: list[int] = None,
        *args,
        **kwargs,
    ) -> tuple[list[float], list[dict[str, Any]]]:
        """
        Compute the reward score for a list of queries and responses.
        """
        assert (
            len(queries)
            == len(responses)
            == len(gold_answers)
            == len(solutions)
            == len(domain_ids)
            == len(cluster_ids)
        )
        # queries contain the task description and problem prompt, need to separate them
        task_descriptions, problem_prompts = self.parse_task_descriptions_and_prompts(queries)
        assert len(task_descriptions) == len(problem_prompts) == len(queries)

        correct_answer_point = 5
        wrong_answer_point = 0

        coin_toss = np.random.choice(["check_answer_only", "reasoning"], p=[0.5, 0.5])

        if coin_toss == "check_answer_only":
            scores = []
            feedbacks = []
            feedback_str = "For this sample, the evaluation is based solely on the alignment between the [Student Response] and the [Reference Solution]."
            for response, gold_answer in zip(responses, gold_answers):
                gold_answer = gold_answer.replace(" ", "").strip()
                student_answer = extract_boxed_content(response)
                student_answer = "" if student_answer is None else student_answer
                rationale = response.replace(student_answer, "").replace("\\boxed{", "").strip()
                # Too short reasoning
                if len(rationale.split()) < 30:
                    scores.append(wrong_answer_point)
                    feedbacks.append(
                        f"<reason>The [Student Response] contains an overly brief rationale. Students are encouraged to provide a fully worked-out solution.</reason> <score>{wrong_answer_point}</score>"
                    )
                # Missing student answer
                elif student_answer is None:
                    scores.append(wrong_answer_point)
                    feedbacks.append(
                        f"<reason>{feedback_str} The answer in [Student Response] was either omitted or not properly formatted with '\\boxed{{}}'</reason> <score>{wrong_answer_point}</score>"
                    )
                # In case we have student answer with reasonable length of rationale,
                else:
                    student_answer = student_answer.replace(" ", "")
                    # Too short student answer
                    if len(student_answer) < 1:
                        scores.append(wrong_answer_point)
                        feedbacks.append(
                            f"<reason>{feedback_str} The answer in [Student Response] was omitted.</reason> <score>{wrong_answer_point}</score>"
                        )
                    # Check if we have correct student answer or not
                    else:
                        gold_answer = gold_answer.replace(" ", "").strip()
                        if is_equiv(gold_answer, student_answer, verbose=False):
                            scores.append(correct_answer_point)
                            feedbacks.append(
                                f"<reason>{feedback_str} The answer provided in [Student Response] matches the reference answer in [Reference Solution]</reason> <score>{correct_answer_point}</score>"
                            )
                        else:
                            scores.append(wrong_answer_point)
                            feedbacks.append(
                                f"<reason>{feedback_str} The answer provided in [Student Response] does not match referenec answer in [Reference Solution]</reason> <score>{wrong_answer_point}</score>"
                            )
            assert len(scores) == len(feedbacks) == len(queries)
        else:
            scores = [None] * len(queries)
            feedbacks = [None] * len(queries)
            llm_state_indices = []
            score_plan_prompts_all = []
            score_execute_prompts_all = []
            for sample_idx, (query, response, solution, dom, clus) in enumerate(
                zip(queries, responses, solutions, domain_ids, cluster_ids)
            ):
                student_answer = extract_boxed_content(response)
                student_answer = "" if student_answer is None else student_answer
                rationale = response.replace(student_answer, "").replace("\\boxed{", "").strip()
                # Too short reasoning
                if len(rationale.split()) < 30:
                    scores[sample_idx] = wrong_answer_point
                    feedbacks[sample_idx] = (
                        f"<reason>The [Student Response] contains an overly brief rationale. Students are encouraged to provide a fully worked-out solution.</reason> <score>{wrong_answer_point}</score>"
                    )
                    continue

                domain_name = idx_to_math_domains[int(dom)]
                plan_input = {
                    "query": query,
                    "solution": solution,
                    "meta_guidelines": self.domain_cluster_guidelines[domain_name][int(clus)],
                }
                score_plan_prompt = self.score_plan_template.render(plan_input)
                score_plan_prompts_all.append(score_plan_prompt)

                score_execute_prompt = self.score_execute_template.render({"response": response})
                score_execute_prompts_all.append(score_execute_prompt)

                llm_state_indices.append(sample_idx)

            inputs = [
                {"plan_prompt": p, "execute_prompt": e}
                for p, e in zip(score_plan_prompts_all, score_execute_prompts_all)
            ]

            states = self.rm_score.run_batch(inputs, backend=self.backend)
            assert len(states) == len(llm_state_indices)

            for sample_idx, s in zip(llm_state_indices, states):
                # per query and response pair,
                try:
                    plan = s["plan"]
                    execute = s["execute"]
                    scores[sample_idx] = int(s["score"])
                    feedback_str = f"<reason>For this specific math problem, the junior instructor developed the [Evaluation Plan] outlined below and then executed it as follows.\n\n{plan}\n\n{execute}\n</reason> <score>{scores[sample_idx]}</score>"
                    feedbacks[sample_idx] = feedback_str
                except Exception as e:
                    print(f"in score-reasoning:{e}")
                    scores[sample_idx] = wrong_answer_point
                    feedbacks[sample_idx] = (
                        f"<reason>For this example, due to the following error, RM-based scoring was not done properly. Error: {str(e)}</reason> <score>{scores[sample_idx]}</score>"
                    )
            assert len(scores) == len(feedbacks) == len(queries)
        if return_evaluations:
            return scores, feedbacks
        return scores


class MetaRewardModelMathReasoning(MetaRewardModel, RewardModelMathReasoning):
    """
    Meta reward model for mathematical reasoning task.
    """

    def __init__(self, reward_model_address: str, experiment_directory: str, **kwargs):
        super().__init__(
            reward_model_address=reward_model_address,
            experiment_directory=experiment_directory,
            **kwargs,
        )
        self.cluster_size = kwargs.get("cluster_size", None)

    @function
    def mrm_prescreen(s, prescreen_prompt: str):
        pass

    @function
    def mrm_analyze_and_refine(s, analyze_prompt: str, refine_prompt: str):
        s += system("You are a skilled mathematician.")
        s += user(analyze_prompt)
        s += assistant(gen("analysis", temperature=0.02, max_tokens=2500, stop=["<EOE>", "</EOE>"]))
        analysis = f"[Start of Meta-Level Analysis]\n{s['analysis'].strip()}\n[End of Meta-Level Analysis]\n"
        s.set_var("analysis", analysis)
        s += user(f"You have provided the following meta-level analysis:\n{analysis}\n")
        s += user(refine_prompt)
        s += assistant(gen("refinement", temperature=0.02, max_tokens=2500, stop=["<EOE>", "</EOE>"]))
        refinement = f"[Start of Meta-Level Guidelines]\n{s['refinement'].strip()}\n[End of Meta-Level Guidelines]\n"
        s.set_var("refinement", refinement)

    @function
    def mrm_merge(s, merge_prompt: str, temperature: float = 0.02):
        pass

    def sample_k_from_three_parts(self, sorted_seq, k: int = 2) -> list[Any]:
        n = len(sorted_seq)
        if n == 0:
            return []
        if n < 3:
            return sorted_seq

        base, rem = divmod(n, 3)
        borders = [0]
        for i in range(3):
            borders.append(borders[-1] + base + (1 if i < rem else 0))

        samples = []
        for start, stop in zip(borders, borders[1:]):
            chunk = sorted_seq[start:stop]
            if len(chunk) < k:
                k = len(chunk)
            samples.extend(random.sample(chunk, k))
        return samples

    def prepare_meta_evaluation_cases_as_prompt(self, evaluation_cases: list[str, Any]) -> str:
        case_outer_form = "[Start of Evaluation Case #{case_idx}]\n{case_inner}\n[End of Evaluation Case #{case_idx}]"
        case_inner_form = "{math_prob}\n{student_response}\n{ref_sol}\n{eval_feedback}\n{eval_score}\n"

        cases_strings = []
        for case_idx, eval_case in enumerate(evaluation_cases, start=1):
            math_prob = (
                f"[Start of Mathematical Problem]\n{eval_case['student_prompt']}\n[End of Mathematical Problem]"
            )
            student_response = (
                f"[Start of Student Response]\n{eval_case['student_generation']}\n[End of Student Response]"
            )
            eval_score = f"[Start of Evaluation Score]\n{eval_case['junior_score']}\n[End of Evaluation Score]"
            eval_feedback = (
                f"[Start of Evaluation Feedback]\n{eval_case['junior_feedback']}\n[End of Evaluation Feedback]"
            )
            ref_sol = f"[Start of Reference Solution]\n{eval_case['solution']}\n[End of Reference Solution]"
            case_inner = case_inner_form.format(
                math_prob=math_prob,
                student_response=student_response,
                ref_sol=ref_sol,
                eval_feedback=eval_feedback,
                eval_score=eval_score,
            )
            case_outer = case_outer_form.format(case_idx=case_idx, case_inner=case_inner)
            cases_strings.append(case_outer)
        return "\n\n".join(cases_strings)

    def meta_evaluate_and_update(
        self,
        batch_index: int,
        return_evaluations: bool = True,
        num_samples: int = None,
        do_prescreening: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Meta-evaluate the model, update the evaluation rubric, and return the analyses.
        """
        domain_cluster_guidelines, domain_cluster_iter_nos = self.read_meta_level_guidelines_and_iter_nos()
        domain_cluster_rollouts = {domain: {} for domain in math_domains_to_idx.keys()}
        for domain in domain_cluster_guidelines.keys():
            for cluster_idx in range(1, self.cluster_size + 1):
                domain_cluster_rollouts[domain][cluster_idx] = []

        rollouts_paths = glob(os.path.join(self.rollouts_directory, f"rollouts-{batch_index}-*.json"))

        for path in rollouts_paths:
            with open(path) as f:
                rollouts = json.load(f)
            queries = [sample["queries"] for sample in rollouts]
            _, _writing_prompts = self.parse_task_descriptions_and_prompts(queries)

            for i, sample in enumerate(rollouts):
                domain_name = idx_to_math_domains[int(sample["domain_ids"])]
                cluster_idx = int(sample["cluster_ids"])
                junior_meta_guideline = domain_cluster_guidelines[domain_name][cluster_idx]
                _input = {
                    "student_prompt": _writing_prompts[i],
                    "student_generation": sample["student_responses"],
                    "junior_meta_guideline": junior_meta_guideline,
                    "junior_score": sample["junior_scores"],
                    "junior_feedback": sample["junior_evaluations"],
                    "solution": sample["solutions"],
                    "max_words": 1000,
                }
                domain_cluster_rollouts[domain_name][cluster_idx].append(_input)

        # Sort by junior scores and select samples for the meta-step
        domain_cluster_meta_samples = {domain: {} for domain in math_domains_to_idx.keys()}
        for domain in domain_cluster_guidelines.keys():
            for cluster_idx in range(1, self.cluster_size + 1):
                domain_cluster_rollouts[domain][cluster_idx] = sorted(
                    domain_cluster_rollouts[domain][cluster_idx], key=lambda sample: sample["junior_score"]
                )
                domain_cluster_meta_samples[domain][cluster_idx] = self.sample_k_from_three_parts(
                    domain_cluster_rollouts[domain][cluster_idx], k=2
                )

        analyze_prompt_all = []
        refine_prompt_all = []
        logs = []

        for domain in domain_cluster_guidelines.keys():
            for cluster_idx in range(1, self.cluster_size + 1):
                meta_sample_input = {
                    "evaluation_cases": self.prepare_meta_evaluation_cases_as_prompt(
                        domain_cluster_meta_samples[domain][cluster_idx]
                    ),
                    "meta_level_guidelines": domain_cluster_guidelines[domain][cluster_idx],
                    "max_words": 1000,
                }
                logs.append(meta_sample_input)
                analyze_prompt = self.analyze_template.render(meta_sample_input)
                analyze_prompt_all.append(analyze_prompt)
                refine_prompt = self.refine_template.render({"max_words": 1000})
                refine_prompt_all.append(refine_prompt)

        inputs = [
            {"analyze_prompt": a_p, "refine_prompt": r_p} for a_p, r_p in zip(analyze_prompt_all, refine_prompt_all)
        ]

        print("Analysis and Refinment step...")
        start_time = time()
        states = self.mrm_analyze_and_refine.run_batch(inputs, backend=self.backend)
        assert len(inputs) == len(states) == (self.cluster_size * len(domain_cluster_guidelines.keys())) == len(logs)
        end_time = time()
        analysis_time = (end_time - start_time) / 60
        print(f"Analysis and Refinement took {analysis_time:.2f} minutes for {len(inputs)} samples.")

        state_index = 0
        for domain in domain_cluster_guidelines.keys():
            for cluster_idx in range(1, self.cluster_size + 1):
                s = states[state_index]
                refinement = s["refinement"]
                logs[state_index]["meta_analysis"] = s["analysis"]
                logs[state_index]["meta_refinement"] = refinement
                logs[state_index]["domain"] = domain
                logs[state_index]["cluster_id"] = cluster_idx
                # update the meta guidelines
                domain_cluster_guidelines[domain][cluster_idx] = refinement
                state_index += 1

        for domain in domain_cluster_guidelines.keys():
            for cluster_idx in range(1, self.cluster_size + 1):
                next_iter_no = domain_cluster_iter_nos[domain][cluster_idx] + 1
                domain_cluster_dir = os.path.join(self.prompts_directory, domain, f"cluster-{cluster_idx}")
                newer_meta_guideline_path = os.path.join(
                    domain_cluster_dir, f"evaluation_rubric_iter_{next_iter_no}.txt"
                )
                with open(newer_meta_guideline_path, "w") as f:
                    f.write(f"{domain_cluster_guidelines[domain][cluster_idx].strip()}\n")
                print(f"{domain}-{cluster_idx}: New version saved to {newer_meta_guideline_path}")

        if return_evaluations:
            assert len(analyze_prompt_all) == len(refine_prompt_all) == len(logs)
            return logs
