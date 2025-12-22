# src/data/preprocessing.py
import os
import pickle
import yaml
import networkx as nx
import numpy as np
import random
from tqdm import tqdm
from src.utils.logger import setup_logger
from src.data.simulation import get_ground_truth_spread

logger = setup_logger()

class DataProcessor:
    def __init__(self, config_path="configs/hyperparams.yaml"):
        # Load config
        # [FIX] Thêm encoding="utf-8" ở đây
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        
        self.raw_dir = self.cfg['dataset']['raw_dir']
        self.processed_dir = self.cfg['dataset']['processed_dir']
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_raw_graph(self):
        """Đọc file edges và nodes, trả về đồ thị Multiplex"""
        logger.info("Loading raw graph data...")
        
        # 1. Load Nodes & Map ID
        node_path = os.path.join(self.raw_dir, self.cfg['dataset']['nodes_file'])
        nodes = []
        
        with open(node_path, 'r', encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue # Bỏ qua dòng trống
                
                # [FIX] Thêm try-except để bỏ qua dòng Header (chứa chữ 'nodeID')
                try:
                    nodes.append(int(parts[0]))
                except ValueError:
                    logger.warning(f"Skipping header or invalid line in nodes file: {parts}")
                    continue
        
        # Map ID từ 1..N sang 0..N-1
        sorted_nodes = sorted(list(set(nodes)))
        self.node_map = {old: new for new, old in enumerate(sorted_nodes)}
        self.num_nodes = len(nodes)
        logger.info(f"Loaded {self.num_nodes} unique nodes.")
        
        # 2. Load Edges vào NetworkX
        self.graphs = {} # {layer_id: nx.DiGraph}
        edge_path = os.path.join(self.raw_dir, self.cfg['dataset']['edges_file'])
        
        with open(edge_path, 'r', encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3: continue
                
                # [FIX] Thêm try-except tương tự cho file Edges
                try:
                    l_id = int(parts[0])
                    u, v = int(parts[1]), int(parts[2])
                except ValueError:
                    # Bỏ qua dòng header nếu có (ví dụ: layerID nodeID...)
                    continue
                
                if u in self.node_map and v in self.node_map:
                    u_mapped, v_mapped = self.node_map[u], self.node_map[v]
                    
                    if l_id not in self.graphs:
                        self.graphs[l_id] = nx.DiGraph()
                        self.graphs[l_id].add_nodes_from(range(self.num_nodes))
                        
                    self.graphs[l_id].add_edge(u_mapped, v_mapped)
        
        # Lưu Graph Object
        graph_out = os.path.join(self.processed_dir, "multiplex_graph.pkl")
        with open(graph_out, "wb") as f:
            pickle.dump({"graphs": self.graphs, "num_nodes": self.num_nodes}, f)
        logger.info(f"Saved graph object to {graph_out}")
        
        # [FIX] Thêm encoding="utf-8"
        with open(edge_path, 'r', encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3: continue
                
                l_id = int(parts[0])
                u, v = int(parts[1]), int(parts[2])
                
                if u in self.node_map and v in self.node_map:
                    u_mapped, v_mapped = self.node_map[u], self.node_map[v]
                    
                    if l_id not in self.graphs:
                        self.graphs[l_id] = nx.DiGraph()
                        self.graphs[l_id].add_nodes_from(range(self.num_nodes))
                        
                    self.graphs[l_id].add_edge(u_mapped, v_mapped)
        
        # Lưu Graph Object
        graph_out = os.path.join(self.processed_dir, "multiplex_graph.pkl")
        with open(graph_out, "wb") as f:
            pickle.dump({"graphs": self.graphs, "num_nodes": self.num_nodes}, f)
        logger.info(f"Saved graph object to {graph_out}")

    def generate_training_data(self):
        """Tạo file .SG chứa cặp (Seed Set, Influence Prob)"""
        num_samples = self.cfg['dataset']['num_samples_train']
        seed_size = self.cfg['dataset']['seed_set_size']
        mc_sims = self.cfg['dataset']['mc_simulations']
        prob = self.cfg['dataset']['propagation_prob']
        
        dataset = []
        logger.info(f"Generating {num_samples} training samples (Simulations)...")
        
        for _ in tqdm(range(num_samples)):
            seed_nodes = random.sample(range(self.num_nodes), seed_size)
            x_vec = np.zeros(self.num_nodes, dtype=np.float32)
            x_vec[seed_nodes] = 1.0
            
            y_vec = get_ground_truth_spread(self.graphs, seed_nodes, self.num_nodes, mc_sims, prob)
            
            dataset.append((x_vec, y_vec))
            
        out_path = os.path.join(self.processed_dir, "train_data.SG")
        with open(out_path, "wb") as f:
            pickle.dump(dataset, f)
        logger.info(f"Dataset saved to {out_path}")