import sys
import os
import yaml
import torch
import random
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.logger import setup_logger

logger = setup_logger()

# --- COPY LẠI HÀM LOAD GRAPH TỪ FILE CŨ ---
def load_multiplex_graph(cfg):
    import pickle
    graph_path = os.path.join(cfg['dataset']['processed_dir'], "multiplex_graph.pkl")
    with open(graph_path, 'rb') as f:
        graph_data = pickle.load(f)
    
    raw_graphs = graph_data['graphs']
    if isinstance(raw_graphs, dict):
        graphs = list(raw_graphs.values())
    else:
        graphs = raw_graphs
    return graphs, graph_data['num_nodes']

def multiplex_independent_cascade(graphs, seeds, mc_steps=20): 
    # Giảm mc_steps xuống 20 để chạy cho nhanh (chấp nhận sai số nhỏ)
    spreads = []
    layers_indegrees = [dict(g.in_degree()) for g in graphs]
    
    for _ in range(mc_steps):
        active_nodes = set(seeds)
        newly_active = set(seeds)
        while newly_active:
            next_active = set()
            for u in newly_active:
                for layer_idx, G in enumerate(graphs):
                    if u not in G: continue
                    for v in G.neighbors(u):
                        if v not in active_nodes and v not in next_active:
                            d_in = layers_indegrees[layer_idx].get(v, 1)
                            prob = 1.0 / d_in if d_in > 0 else 0
                            if random.random() < prob:
                                next_active.add(v)
            active_nodes.update(next_active)
            newly_active = next_active
        spreads.append(len(active_nodes))
    return np.mean(spreads)

def augment_training_data():
    # 1. Config
    config_path = "configs/hyperparams.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # 2. Seed Set "Cứng đầu" từ log của bạn
    base_seeds = [500, 1288, 1699, 3033, 3602, 2421, 997, 833, 3542, 3538, 
                  1120, 2345, 2577, 663, 389, 1030, 2970, 2269, 3196, 3661, 
                  3679, 400, 3334, 1506, 3225, 2105, 350, 3773, 66, 1477, 
                  2614, 722, 3024, 2863, 715, 1547, 2033, 191, 2232, 1649, 
                  2620, 1221, 1432, 1873, 1534, 2663, 1633, 2621, 1026, 1146]
    
    logger.info("Loading Graph...")
    graphs, num_nodes = load_multiplex_graph(cfg)
    all_nodes = list(range(num_nodes))
    
    new_data = []
    num_samples = 50 # Tạo 50 mẫu dữ liệu mới
    
    logger.info(f"Generating {num_samples} augmented samples (Self-Correction)...")
    
    for i in tqdm(range(num_samples)):
        # Chiến thuật: Giữ 80% seed cũ, thay 20% bằng random node
        # Để dạy model biết lân cận của đỉnh núi trông như thế nào
        current_seeds = base_seeds.copy()
        
        # Mutation: Thay thế 10 node ngẫu nhiên
        num_mutations = 10
        for _ in range(num_mutations):
            remove_idx = random.randint(0, len(current_seeds)-1)
            current_seeds.pop(remove_idx)
            current_seeds.append(random.choice(all_nodes))
            
        # Tính nhãn THẬT (Ground Truth)
        # Đây là bước quan trọng nhất: Dạy model thực tế
        true_spread = multiplex_independent_cascade(graphs, current_seeds)
        
        # Format dữ liệu để lưu: (binary_vector_x, scalar_y)
        x_binary = torch.zeros(num_nodes)
        x_binary[current_seeds] = 1.0
        
        # Normalize y nếu cần (ở đây giữ nguyên raw spread)
        y_value = torch.tensor([true_spread], dtype=torch.float)
        
        new_data.append((x_binary, y_value))

    # 3. Lưu vào file mới
    save_path = os.path.join(cfg['dataset']['processed_dir'], "augmented_data.SG")
    logger.info(f"Saving {len(new_data)} samples to {save_path}...")
    torch.save(new_data, save_path)
    
    # 4. Hợp nhất với data cũ (Optional)
    # Ở đây mình khuyên train trên data mới trước để Fine-tune cực mạnh (Shock therapy)
    logger.info("DONE! Now run training with this new dataset.")

if __name__ == "__main__":
    augment_training_data()