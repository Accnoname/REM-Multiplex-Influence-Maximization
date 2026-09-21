import os
import pickle
import yaml
import logging
import networkx as nx
import random
from tqdm import tqdm

from src.data.simulation import get_ground_truth_spread

log = logging.getLogger("REM")


class DataProcessor:
    def __init__(self, config_path="configs/hyperparams.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.raw_dir = self.cfg["dataset"]["raw_dir"]
        self.processed_dir = self.cfg["dataset"]["processed_dir"]
        os.makedirs(self.processed_dir, exist_ok=True)
        self.graphs = {}
        self.num_nodes = 0

    def load_raw_graph(self):
        node_path = os.path.join(self.raw_dir, self.cfg["dataset"]["nodes_file"])
        nodes = []
        with open(node_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    nodes.append(int(parts[0]))
                except ValueError:
                    continue  # skip header

        sorted_nodes = sorted(set(nodes))
        self.node_map = {old: new for new, old in enumerate(sorted_nodes)}
        self.num_nodes = len(sorted_nodes)
        log.info(f"Loaded {self.num_nodes} unique nodes.")

        edge_path = os.path.join(self.raw_dir, self.cfg["dataset"]["edges_file"])
        with open(edge_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                try:
                    l_id, u, v = int(parts[0]), int(parts[1]), int(parts[2])
                except ValueError:
                    continue
                if u in self.node_map and v in self.node_map:
                    um, vm = self.node_map[u], self.node_map[v]
                    if l_id not in self.graphs:
                        self.graphs[l_id] = nx.DiGraph()
                        self.graphs[l_id].add_nodes_from(range(self.num_nodes))
                    self.graphs[l_id].add_edge(um, vm)

        graph_out = os.path.join(self.processed_dir, "multiplex_graph.pkl")
        with open(graph_out, "wb") as f:
            pickle.dump({"graphs": self.graphs, "num_nodes": self.num_nodes}, f)
        log.info(f"Saved graph → {graph_out} ({len(self.graphs)} layers)")

    def generate_training_data(self):
        num_samples = self.cfg["dataset"]["num_samples_train"]
        seed_size = self.cfg["dataset"]["seed_set_size"]
        mc_sims = self.cfg["dataset"]["mc_simulations"]

        dataset = []
        log.info(f"Generating {num_samples} training samples...")

        import numpy as np
        for _ in tqdm(range(num_samples)):
            seeds = random.sample(range(self.num_nodes), seed_size)
            x = np.zeros(self.num_nodes, dtype=np.float32)
            x[seeds] = 1.0
            y = get_ground_truth_spread(self.graphs, seeds, self.num_nodes, mc_sims)
            dataset.append((x, y))

        out = os.path.join(self.processed_dir, "train_data.SG")
        with open(out, "wb") as f:
            pickle.dump(dataset, f)
        log.info(f"Saved {num_samples} samples → {out}")