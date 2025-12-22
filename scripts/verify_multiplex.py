import sys
import os
import yaml
import torch
import numpy as np
import random
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.logger import setup_logger

logger = setup_logger()

def load_multiplex_graph(cfg):
    import pickle
    graph_path = os.path.join(cfg['dataset']['processed_dir'], "multiplex_graph.pkl")
    with open(graph_path, 'rb') as f:
        graph_data = pickle.load(f)
    
    # --- [ENGINEER FIX] ---
    # Kiểm tra xem dữ liệu là Dict hay List để xử lý đúng
    raw_graphs = graph_data['graphs']
    
    if isinstance(raw_graphs, dict):
        # Nếu là dict {id: Graph}, ta chỉ lấy phần Graph (values)
        graphs = list(raw_graphs.values())
        logger.info("Detected Dictionary graph storage. Converted to List.")
    else:
        graphs = raw_graphs
    # ----------------------
    
    return graphs

def multiplex_independent_cascade(graphs, seeds, mc_steps=50):
    """
    Mô phỏng lan truyền đa lớp (Multiplex Weighted Cascade).
    Cơ chế: Node u bị nhiễm ở layer bất kỳ -> u bị nhiễm toàn cục -> u lây lan trên TẤT CẢ các layer.
    """
    spreads = []
    
    # Pre-compute in-degrees cho từng layer để tối ưu tốc độ
    layers_indegrees = [dict(g.in_degree()) for g in graphs]
    
    for _ in tqdm(range(mc_steps), desc="Multiplex MC Simulation"):
        active_nodes = set(seeds) # Tập các node đã bị nhiễm (trên bất kỳ layer nào)
        newly_active = set(seeds) # Các node vừa mới bị nhiễm ở bước trước
        
        while newly_active:
            next_active = set()
            
            # Với mỗi node vừa bị nhiễm, nó sẽ cố lây lan trên TẤT CẢ các layer
            for u in newly_active:
                
                # Duyệt qua từng layer (Facebook, Twitter,...)
                for layer_idx, G in enumerate(graphs):
                    if u not in G: continue
                    
                    # Thử lây cho hàng xóm trong layer này
                    for v in G.neighbors(u):
                        if v not in active_nodes and v not in next_active:
                            # Logic Weighted Cascade: p = 1 / in_degree(v) tại layer này
                            d_in = layers_indegrees[layer_idx].get(v, 1)
                            prob = 1.0 / d_in if d_in > 0 else 0
                            
                            if random.random() < prob:
                                next_active.add(v)
            
            # Cập nhật danh sách nhiễm
            active_nodes.update(next_active)
            newly_active = next_active
            
        spreads.append(len(active_nodes))
        
    return np.mean(spreads)

def verify_multiplex():
    # 1. Load Config
    config_path = "configs/hyperparams.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. Seed Set (Kết quả tốt nhất của bạn)
    optimal_seeds = [500, 1288, 1699, 3033, 3602, 2421, 997, 833, 3542, 3538, 
                     2345, 1120, 663, 2577, 389, 1030, 2970, 2269, 3661, 3196, 
                     3679, 400, 3334, 1506, 3225, 350, 2105, 3773, 1477, 66, 
                     722, 2614, 3024, 2863, 715, 1547, 191, 2033, 2232, 1649, 
                     2620, 1221, 1873, 1432, 1534, 2663, 2621, 1633, 1026, 299]
    
    logger.info(f"Verifying Seed Set Size: {len(optimal_seeds)}")
    
    # 3. Load Multiplex Graph
    graphs = load_multiplex_graph(cfg)
    logger.info(f"Multiplex Graph Loaded: {len(graphs)} layers.")
    
    # 4. Run Simulation
    logger.info(f"Running Multiplex Monte Carlo...")
    actual_spread = multiplex_independent_cascade(graphs, optimal_seeds, mc_steps=50)
    
    logger.info("="*30)
    logger.info(f"REM Paper Benchmark : ~426.7")
    logger.info(f"Single Layer Result : 199.0")
    logger.info(f"Multiplex Result    : {actual_spread:.2f}")
    logger.info("="*30)

if __name__ == "__main__":
    verify_multiplex()