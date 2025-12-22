# TODO: Implement logic here
# src/data/simulation.py
import random
import numpy as np
from tqdm import tqdm

def run_IC_simulation(graphs, seed_set, prob=0.1):
    """
    Mô phỏng 1 lần lan truyền IC trên Multiplex.
    Args:
        graphs: Dict {layer_id: nx.DiGraph}
        seed_set: List node index
    """
    # Active set bao gồm seed nodes ban đầu
    active_set = set(seed_set)
    newly_active = list(active_set)
    
    while newly_active:
        next_newly_active = set()
        
        # Duyệt qua các node mới bị nhiễm
        for u in newly_active:
            # Lan truyền trên TẤT CẢ các layer (Multiplex logic)
            for layer_id, G in graphs.items():
                if G.has_node(u):
                    neighbors = list(G.successors(u))
                    for v in neighbors:
                        if v not in active_set and v not in next_newly_active:
                            # Tung xúc xắc lây nhiễm
                            if random.random() < prob:
                                next_newly_active.add(v)
        
        if not next_newly_active:
            break
            
        active_set.update(next_newly_active)
        newly_active = list(next_newly_active)
        
    return active_set

def get_ground_truth_spread(graphs, seed_set, num_nodes, mc_sims=50, prob=0.1):
    """
    Chạy Monte Carlo nhiều lần để lấy xác suất nhiễm trung bình.
    Output: Vector xác suất y (dùng để train PMoE).
    """
    infection_counts = np.zeros(num_nodes)
    
    # Chạy vòng lặp Monte Carlo [cite: 93]
    for _ in range(mc_sims):
        infected_nodes = run_IC_simulation(graphs, seed_set, prob)
        for node in infected_nodes:
            infection_counts[node] += 1
            
    # Chuẩn hóa về xác suất [0, 1]
    return infection_counts / mc_sims