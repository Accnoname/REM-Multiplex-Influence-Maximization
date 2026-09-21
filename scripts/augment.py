import sys
import os
import yaml
import random
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
from src.data.dataset import load_graph
from src.data.simulation import get_ground_truth_spread

log = logging.getLogger("REM")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Best seeds from latest inference (k=50)
BASE_SEEDS = [
    66, 3225, 350, 2421, 1699, 3538, 1221, 1649, 2577, 1633,
    1030, 2345, 1120, 3602, 3033, 1477, 3679, 997, 833, 3542,
    389, 2269, 3661, 3196, 400, 3334, 1506, 2105, 3773, 722,
    2614, 3024, 2863, 715, 1547, 191, 2033, 2232, 2620, 1873,
    1432, 1534, 2663, 2621, 1026, 299, 500, 1288, 304, 276,
]


def augment():
    with open("configs/hyperparams.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _, _, graph_list = load_graph(cfg)
    num_nodes = cfg["dataset"].get("num_nodes") or len(BASE_SEEDS) * 80  # fallback
    # Get real num_nodes from graph
    import pickle
    with open(os.path.join(cfg["dataset"]["processed_dir"], "multiplex_graph.pkl"), "rb") as f:
        gd = pickle.load(f)
    num_nodes = gd["num_nodes"]
    all_nodes = list(range(num_nodes))

    num_samples = 50
    data = []
    log.info(f"Generating {num_samples} augmented samples...")

    for _ in tqdm(range(num_samples)):
        seeds = BASE_SEEDS.copy()
        for _ in range(10):  # mutate 10 nodes
            seeds.pop(random.randrange(len(seeds)))
            seeds.append(random.choice(all_nodes))

        # ponytail: mc_sims=20 for speed; raise to 50 if label noise matters
        spread_vec = get_ground_truth_spread(graph_list, seeds, num_nodes, mc_sims=20)
        x = torch.zeros(num_nodes)
        x[seeds] = 1.0
        # y = per-node infection probability vector [num_nodes], matches .SG paper format
        y = torch.tensor(spread_vec, dtype=torch.float32)
        data.append((x, y))

    save_path = os.path.join(cfg["dataset"]["processed_dir"], "augmented_data.SG")
    torch.save(data, save_path)
    log.info(f"Saved {len(data)} samples to {save_path}")


if __name__ == "__main__":
    augment()
