import os
import numpy as np
from itertools import combinations
from polyleven import levenshtein
from tabulate import tabulate
from typing import List


def edit_distance(seq1, seq2):
    return levenshtein(seq1, seq2)


def measure_diversity(seqs: List[str]):
    dists = []
    for pair in combinations(seqs, 2):
        dists.append(edit_distance(*pair))
    return np.mean(dists)


def measure_distwt(seqs: List[str], wt: str):
    dists = []
    for seq in seqs:
        dists.append(edit_distance(seq, wt))
    return np.mean(dists)


def measure_novelty(seqs: List[str], train_seqs: List[str]):
    all_novelty = []
    for seq in seqs:
        min_dist = 1e9
        for known in train_seqs:
            dist = edit_distance(seq, known)
            if dist == 0:
                all_novelty.append(dist)
                break
            elif dist < min_dist:
                min_dist = dist
        all_novelty.append(min_dist)
    return np.mean(all_novelty)


def remove_duplicates(seqs: List[str], scores: List[float], return_idx: bool = False):
    new_seqs = []
    new_scores = []
    ids = []
    for idx, (seq, score) in enumerate(zip(seqs, scores)):
        if seq in new_seqs:
            continue
        else:
            new_seqs.append(seq)
            new_scores.append(score)
            ids.append(idx)
    return new_seqs, new_scores, ids if return_idx else None


def parse_module_name_from_path(path: str) -> str:
    """Designed specifically for config files inside configs folder."""
    names = os.path.splitext(path)[0].split(os.sep)[-3:]
    module = ".".join(names)
    return module


def parse_dict_from_module(module):
    start = False
    module_dict = module.__dict__
    newdict = {}
    for key in module_dict.keys():
        if start:
            newdict[key] = module_dict[key]
        if key != "os":
            continue
        else:
            start = True

    return newdict


def print_stats(sequences: List[str],
                fitnesses: List[float],
                wt_seq: str,
                oracle_fitness: List[float] = None,
                return_stats: bool = True):
    seqs, scores, ids = remove_duplicates(sequences, fitnesses, return_idx=True)
    if oracle_fitness is not None:
        oracle_scores = [oracle_fitness[i] for i in ids]
    length = f"{len(seqs)}/{len(sequences)}"
    max_fitness = round(max(scores), 3)
    mean_fitness = round(np.mean(scores), 3)
    oracle_max_fit = round(max(oracle_scores), 3) if oracle_fitness is not None else "N/A"
    oracle_mean_fit = round(np.mean(oracle_scores), 3) if oracle_fitness is not None else "N/A"
    diversity = round(measure_diversity(sequences), 3)
    dist_wt = round(measure_distwt(sequences, wt_seq), 3)
    table = tabulate(
        [[length, mean_fitness, oracle_mean_fit, max_fitness,
          oracle_max_fit, diversity, dist_wt]],
        headers=["# Unique", "Mean fitness", "Orc. mean fitness", "Max fitness",
                 "Orc. max fitness", "Diversity", "dist(WT)"]
    )
    print(table)
    return table if return_stats else None
