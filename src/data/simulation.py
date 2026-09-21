# src/data/simulation.py
import random
import numpy as np
from tqdm import tqdm

def run_IC_simulation(graphs, seed_set, prob=None, model="WC"):
    """
    Mô phỏng 1 lần lan truyền trên Multiplex Network.
    Hỗ trợ mô hình Weighted Cascade (WC: p = 1 / d_in) hoặc IC (prob cố định).
    Cơ chế Overlapping Activation: khi u bị kích hoạt, bản sao trên tất cả các layer
    của u đều được kích hoạt và lan truyền.
    """
    if isinstance(graphs, dict):
        graph_list = list(graphs.values())
    else:
        graph_list = graphs

    active_set = set(seed_set)
    newly_active = set(seed_set)

    # Precompute in-degrees with caching
    layers_indegrees = []
    for G in graph_list:
        if not hasattr(G, "_indegrees"):
            if hasattr(G, "in_degree"):
                G._indegrees = dict(G.in_degree())
            else:
                G._indegrees = dict(G.degree())
        layers_indegrees.append(G._indegrees)

    while newly_active:
        next_newly_active = set()

        for u in newly_active:
            for layer_idx, G in enumerate(graph_list):
                if not G.has_node(u):
                    continue

                if hasattr(G, "successors"):
                    neighbors = G.successors(u)
                else:
                    neighbors = G.neighbors(u)

                for v in neighbors:
                    if v not in active_set and v not in next_newly_active:
                        if model == "WC":
                            d_in = layers_indegrees[layer_idx].get(v, 1)
                            p = 1.0 / d_in if d_in > 0 else 0.1
                        else:
                            p = prob if prob is not None else 0.1

                        if random.random() < p:
                            next_newly_active.add(v)

        if not next_newly_active:
            break

        active_set.update(next_newly_active)
        newly_active = next_newly_active

    return active_set

def get_ground_truth_spread(graphs, seed_set, num_nodes, mc_sims=50, prob=None, model="WC"):
    """
    Chạy Monte Carlo nhiều lần để lấy xác suất nhiễm trung bình cho từng node.
    Output: Vector xác suất y (kích thước num_nodes) dùng để train PMoE.
    """
    infection_counts = np.zeros(num_nodes, dtype=np.float32)

    for _ in range(mc_sims):
        infected_nodes = run_IC_simulation(graphs, seed_set, prob=prob, model=model)
        for node in infected_nodes:
            infection_counts[node] += 1.0

    return infection_counts / mc_sims