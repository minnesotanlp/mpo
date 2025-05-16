import json
import multiprocessing as mp
import os
import pickle

import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm

from trl.extras import mpo


def _compute_rouge(gen):
    clean_summary = query_to_clean_summ[gen["query"]]
    score = scorer.score(clean_summary, gen["model_response"])
    return score


def compute_rouge(path):
    generations = []
    with open(path) as f:
        for line in f:
            generations.append(json.loads(line))
    with mp.Pool(mp.cpu_count()) as p:
        scores = list(tqdm(p.imap(_compute_rouge, generations, chunksize=2), total=len(generations)))
    score_types = {"rouge1": [], "rouge2": [], "rougeL": [], "rougeLsum": []}

    for score in scores:
        for s_t, values in score_types.items():
            values.append(score[s_t].fmeasure)
    for s_t, values in score_types.items():
        print(f"\t{s_t}: {np.mean(values) * 100:.2f}")


if __name__ == "__main__":
    query_to_clean_summ_path = os.path.join(
        os.path.dirname(mpo.__file__), "corpora", "BillSum", "query_to_clean_summ.pkl"
    )
    with open(query_to_clean_summ_path, "rb") as f:
        query_to_clean_summ = pickle.load(f)

    gen_dir = "results/policy-1.5b/generations/summarization"
    model_names = ["32b_32b", "autoprompt-32b", "iter0-32b", "base"]

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)

    for model_n in model_names:
        path = os.path.join(gen_dir, f"{model_n}.test.generations.jsonl")
        print(f"{model_n}:")
        compute_rouge(path)
