import json
import os
import random
import sys
from glob import glob
from pprint import pprint

import matplotlib.pyplot as plt
import openai
import regex as re
from dotenv import load_dotenv
from natsort import natsorted
from rich.progress import track


load_dotenv()  # take environment variables from .env.

exp_name = sys.argv[1]  # "mpo_variations" "rm_32" "rm_72" "32b_32bvs32b_72b" ""72b_32bvs72b_72b""
num_matches = 2000
task_name = "summarization"  # "essay_writing"
print(f"exp_name is: {exp_name}")

if exp_name == "rm_32":
    model_names_to_annon = {
        "32b_72b": "ModelA",
        "autoprompt-32b": "ModelB",
        "expert-32b": "ModelC",
        "base-1.5b": "ModelD",
        "iter0-32b": "ModelE",
    }
elif exp_name == "rm_72":
    model_names_to_annon = {
        "72b_72b": "ModelA",
        "autoprompt-72b": "ModelB",
        "expert-72b": "ModelC",
        "base-1.5b": "ModelD",
        "iter0-72b": "ModelE",
    }
elif exp_name == "mpo_variations":
    model_names_to_annon = {
        "32b_32b": "ModelA",
        "32b_72b": "ModelB",
        "72b_32b": "ModelC",
        "72b_72b": "ModelD",
    }
elif exp_name == "32b_32bvs32b_72b":
    model_names_to_annon = {
        "32b_32b": "ModelA",
        "32b_72b": "ModelB",
    }
elif exp_name == "72b_32bvs72b_72b":
    model_names_to_annon = {
        "72b_32b": "ModelA",
        "72b_72b": "ModelB",
    }
elif exp_name == "mpo_vs_oracle":
    model_names_to_annon = {
        "32b_72b": "ModelB",
        "72b_72b": "ModelD",
        "oracle-32b": "ModelI",
        "oracle-72b": "ModelJ",
    }
elif exp_name == "summarization":
    model_names_to_annon = {"32b_32b": "ModelA", "autoprompt-32b": "ModelB", "iter0-32b": "ModelC", "base": "ModelD"}

annon_to_model_names = {v: k for k, v in model_names_to_annon.items()}


def load_data(jsonl_paths, separation_regex):
    data_dict = {}
    for model_name, path in jsonl_paths.items():
        entries = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line_data = json.loads(line.strip())
                m = re.search(separation_regex, line_data["query"], re.MULTILINE | re.DOTALL)
                query = m.group(2).strip()
                entries.append((query, line_data["model_response"]))
        annon_name = model_names_to_annon[model_name]
        data_dict[annon_name] = entries
    return data_dict


def update_elo(rating_a, rating_b, score_a, k=32, BASE=10, SCALE=400):
    """
    Updates and returns the Elo rating for A and B after a match.

    Inputs:
        rating_a, rating_b: Current Elo ratings of A and B.
        score_a: 1 if A wins, 0 if B wins.
        k: K-factor for controlling rating volatility.
    Returns:
        new_rating_a, new_rating_b
    """
    ea = 1 / (1 + BASE ** ((rating_b - rating_a) / SCALE))
    eb = 1 / (1 + BASE ** ((rating_a - rating_b) / SCALE))

    new_rating_a = rating_a + k * (score_a - ea)
    new_rating_b = rating_b + k * ((1 - score_a) - eb)
    return new_rating_a, new_rating_b


def call_openai_api_essay_writing(prompt, temperature=0.0, model="gpt-3.5-turbo"):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an impartial and insightful judge. Evaluate and determine which written essay best responds to a given writing prompt, considering factors such as clarity, coherence, depth of argument, and overall effectiveness.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        stop="<EOE>",
    )
    return completion.choices[0].message.content


def call_openai_api_summarization(prompt, temperature=0.0, model="gpt-3.5-turbo"):
    """
    Calls the OpenAI model to evaluate a prompt and returns
    the response text.
    """
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an impartial and discerning evaluator. Assess which written summary most effectively represents the given government bill, taking into account factors such as conciseness, accuracy, and faithfulness to the original content.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        stop="<EOE>",
    )
    return completion.choices[0].message.content


def build_comparison_prompt_essay_writing(query, model_a_name, model_a_response, model_b_name, model_b_response):
    """
    Constructs a user prompt that includes the query and the two candidate responses,
    asking the judge (ChatGPT) to pick which model's response is better.
    """
    prompt = f"""
Writing Prompt:
{query}

Response from {model_a_name}:
{model_a_response}

Response from {model_b_name}:
{model_b_response}

Which response is better overall? Please simply reply with either '<winner>{model_a_name}</winner>' or '<winner>{model_b_name}</winner>', followed by <EOE>.
"""
    return prompt.strip()


def build_comparison_prompt_summarization(query, model_a_name, model_a_response, model_b_name, model_b_response):
    """
    Constructs a user prompt that includes the query and the two candidate responses,
    asking the judge (ChatGPT) to pick which model's response is better.
    """
    prompt = f"""
Government Bill:
{query}

Summary from {model_a_name}:
{model_a_response}

Summary from {model_b_name}:
{model_b_response}

Which summary is better overall? Please simply reply with either '<winner>{model_a_name}</winner>' or '<winner>{model_b_name}</winner>', followed by <EOE>.
"""
    return prompt.strip()


def judge_responses(
    task_name, query, model_a_name, model_a_response, model_b_name, model_b_response, judge_model="gpt-4o"
):
    """
    Sends the comparison prompt to OpenAI and returns the winner model name.
    """
    if task_name == "essay_writing":
        build_comparison_prompt = build_comparison_prompt_essay_writing
        call_openai_api = call_openai_api_essay_writing
    elif task_name == "summarization":
        build_comparison_prompt = build_comparison_prompt_summarization
        call_openai_api = call_openai_api_summarization
    prompt = build_comparison_prompt(query, model_a_name, model_a_response, model_b_name, model_b_response)
    judge_reply = call_openai_api(prompt, model=judge_model).strip()

    # Simple logic to detect the winner from the judge's reply. Adjust as needed.
    winner_rgx = r"<winner>(.+)</winner>"
    m = re.search(winner_rgx, judge_reply.lower(), re.MULTILINE | re.DOTALL)
    try:
        winner = m.group(1).strip()
        if model_a_name.lower() in winner:
            return model_a_name
        elif model_b_name.lower() in winner:
            return model_b_name
        else:
            # Fallback if the answer is ambiguous; could choose random or skip
            return None
    except:
        return None


def run_elo_simulation(task_name, data_dict, num_matches=1000, k_factor=4, judge_model="gpt-3.5-turbo"):
    """
    Runs the Elo simulation. Randomly samples pairs from the loaded data,
    calls the judge, updates Elo ratings, and returns the final scores.

    Inputs:
        data_dict: Dictionary of model_name -> list of (query, model_response).
        num_matches: Number of pairwise comparisons to run.
        k_factor: K-factor for Elo updates.
        judge_model: Which OpenAI model to use for judgments.
    Returns:
        elo_scores: Dictionary of model_name -> final Elo rating.
    """
    # Initialize Elo scores
    elo_scores = {model: 1000 for model in data_dict.keys()}
    models = list(data_dict.keys())

    results = []

    missed_cnt = 0
    for _ in track(range(num_matches), description="Judging...", total=num_matches):
        # Sample two distinct models
        model_a_name, model_b_name = random.sample(models, 2)

        # Pick a random query index (assuming all have 4096 entries aligned)
        q_idx = random.randint(0, len(data_dict[model_a_name]) - 1)
        query_a, model_a_response = data_dict[model_a_name][q_idx]
        query_b, model_b_response = data_dict[model_b_name][q_idx]
        assert query_a == query_b

        winner = judge_responses(
            task_name, query_a, model_a_name, model_a_response, model_b_name, model_b_response, judge_model=judge_model
        )

        if winner is None:
            missed_cnt += 1
            continue

        if winner == model_a_name:
            new_a, new_b = update_elo(elo_scores[model_a_name], elo_scores[model_b_name], score_a=1, k=k_factor)
        else:
            new_a, new_b = update_elo(elo_scores[model_a_name], elo_scores[model_b_name], score_a=0, k=k_factor)

        elo_scores[model_a_name] = new_a
        elo_scores[model_b_name] = new_b

        results.append(
            {
                "query": query_a,
                "model_a": annon_to_model_names[model_a_name],
                "model_b": annon_to_model_names[model_b_name],
                "winner": annon_to_model_names[winner],
                "model_a_response": model_a_response,
                "model_b_response": model_b_response,
            }
        )

    for _ in track(range(missed_cnt), description="Re-Judging...", total=missed_cnt):
        # Sample two distinct models
        model_a_name, model_b_name = random.sample(models, 2)

        # Pick a random query index (assuming all have 4096 entries aligned)
        q_idx = random.randint(0, len(data_dict[model_a_name]) - 1)
        query_a, model_a_response = data_dict[model_a_name][q_idx]
        query_b, model_b_response = data_dict[model_b_name][q_idx]
        assert query_a == query_b

        winner = judge_responses(
            query_a, model_a_name, model_a_response, model_b_name, model_b_response, judge_model=judge_model
        )

        if winner is None:
            continue

        if winner == model_a_name:
            new_a, new_b = update_elo(elo_scores[model_a_name], elo_scores[model_b_name], score_a=1, k=k_factor)
        else:
            new_a, new_b = update_elo(elo_scores[model_a_name], elo_scores[model_b_name], score_a=0, k=k_factor)

        elo_scores[model_a_name] = new_a
        elo_scores[model_b_name] = new_b

        results.append(
            {
                "query": query_a,
                "model_a": annon_to_model_names[model_a_name],
                "model_b": annon_to_model_names[model_b_name],
                "winner": annon_to_model_names[winner],
                "model_a_response": model_a_response,
                "model_b_response": model_b_response,
            }
        )

    return elo_scores, results


def save_elo_scores_to_json(elo_scores, output_path="elo_scores.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(elo_scores, f, indent=2)


def plot_elo_scores(elo_scores, output_path):
    """
    Creates a bar chart of the Elo scores using matplotlib.
    """
    sorted_scores = sorted(elo_scores.items(), key=lambda x: x[1], reverse=True)
    models = [x[0] for x in sorted_scores]
    scores = [x[1] for x in sorted_scores]

    plt.rcParams.update({"font.size": 16})
    plt.figure(figsize=(10, 6))
    plt.bar(models, scores, color="skyblue")
    plt.title("Elo Ratings of Language Models")
    plt.xlabel("Model")
    plt.ylabel("Elo Rating")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)


def run_sim(task_name, iteration, num_matches, generation_dir, output_dir, exp_name, separation_regex, judge_model):
    print(f"{task_name}: Running Elo simulation for iteration {iteration}...")

    jsonl_paths = {k: f"{generation_dir}/{k}.test.generations.jsonl" for k in model_names_to_annon.keys()}

    data_dict = load_data(jsonl_paths, separation_regex)
    one_key = list(data_dict.keys())[0]
    print(f"len(data_dict): {len(data_dict)}")
    print(f"len(data_dict[{one_key}]): {len(data_dict[one_key])}")

    final_elo_scores, elo_results = run_elo_simulation(
        task_name,
        data_dict,
        num_matches=num_matches,
        k_factor=4,
        judge_model=judge_model,
    )

    final_elo_scores = {annon_to_model_names[k]: v for k, v in final_elo_scores.items()}

    elo_scores_path = f"{output_dir}/{exp_name}-{iteration}.scores.json"
    save_elo_scores_to_json(final_elo_scores, output_path=elo_scores_path)

    with open(elo_scores_path) as f:
        final_elo_scores = json.load(f)

    plot_elo_scores(final_elo_scores, output_path=f"{elo_scores_path[:-5]}_plot.pdf")

    with open(f"{output_dir}/{exp_name}-{iteration}.details.json", "w") as f:
        json.dump(elo_results, f, indent=2)


def merge_results(output_dir: str, exp_name: str):
    details_paths = natsorted(glob(os.path.join(output_dir, f"{exp_name}-*.details.json")))
    scores_paths = natsorted(glob(os.path.join(output_dir, f"{exp_name}-*.scores.json")))
    all_details = []
    for d_p in details_paths:
        with open(d_p) as f:
            all_details.extend(json.load(f))
    print(f"Total number of details: {len(all_details)}")
    all_scores = []
    for s_p in scores_paths:
        with open(s_p) as f:
            scores = json.load(f)
        all_scores.append(scores)
    pprint(all_scores)
    # compute mean of all scores by their keys
    mean_scores = {}
    mean_stds = {}
    for k in all_scores[0].keys():
        mean_scores[k] = sum([s[k] for s in all_scores]) / len(all_scores)
        mean_stds[k] = (sum([(s[k] - mean_scores[k]) ** 2 for s in all_scores]) / len(all_scores)) ** 0.5
    pprint(f"Mean scores: {mean_scores}")
    pprint(f"Mean stds: {mean_stds}")

    merged_details_path = os.path.join(output_dir, f"{exp_name}-merged.details.json")
    merged_scores_path = os.path.join(output_dir, f"{exp_name}-merged.scores.json")
    with open(merged_details_path, "w") as f:
        json.dump(all_details, f, indent=2)
    all_stats = {"mean elo": mean_scores, "mean stds": mean_stds}
    with open(merged_scores_path, "w") as f:
        json.dump(all_stats, f, indent=2)


if __name__ == "__main__":
    client = openai.OpenAI(api_key=os.environ["OPENAI_KEY"])
    judge_model = "gpt-4o"

    generation_dir = f"results/policy-1.5b/generations/{task_name}"
    output_dir = f"results/policy-1.5b/elo_scores/{task_name}"
    if task_name == "essay_writing":
        separation_regex = r"user(.+?)Instructions:(.+?)Your Writing:"
    elif task_name == "summarization":
        separation_regex = r"You are a helpful assistant\.\nuser(.+?)Bill:\n```(.+?)```"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(1, 6):
        run_sim(task_name, i, num_matches, generation_dir, output_dir, exp_name, separation_regex, judge_model)
    merge_results(output_dir, exp_name)
