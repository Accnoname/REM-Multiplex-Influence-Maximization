import sys
import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import random
from tqdm import tqdm

# Import modules của dự án
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.seed2vec import Seed2Vec
from src.models.pmoe import PMoE
from src.utils.logger import setup_logger
from scripts.inference_robust import robust_inference

# Setup Logger
logger = setup_logger()

# ==========================================
# 1. HÀM CHẠY MONTE CARLO (CHUẨN WEIGHTED CASCADE)
# ==========================================
def run_stable_monte_carlo(seeds, graph_data, n_simulations=200):
    """
    Chạy Monte Carlo N lần và lấy trung bình.
    Sử dụng mô hình Weighted Cascade: p(u,v) = 1 / in_degree(v).
    """
    raw_layers = graph_data['graphs']
    
    # [FIX] Chuyển đổi Dictionary thành List các đồ thị
    if isinstance(raw_layers, dict):
        layers = [raw_layers[k] for k in sorted(raw_layers.keys())]
    else:
        layers = raw_layers
        
    total_spreads = []

    # Chạy N lần mô phỏng
    for _ in tqdm(range(n_simulations), desc="Monte Carlo Simulation", leave=False):
        active_nodes = set(seeds)
        newly_active = set(seeds)
        
        while newly_active:
            next_newly_active = set()
            
            # 1. Lan truyền trong từng lớp (Intra-layer)
            for G_layer in layers:
                for node in newly_active:
                    if node not in G_layer: continue
                    
                    neighbors = list(G_layer.neighbors(node))
                    for neighbor in neighbors:
                        if neighbor not in active_nodes:
                            # [REVISED] WEIGHTED CASCADE MODEL
                            # Logic: Xác suất lây nhiễm tỷ lệ nghịch với số bạn bè
                            try:
                                # Lấy bậc vào (in-degree)
                                if G_layer.is_directed():
                                    d_in = G_layer.in_degree(neighbor)
                                else:
                                    d_in = G_layer.degree(neighbor)
                                
                                # Tính trọng số p = 1/d_in
                                if d_in > 0:
                                    weight = 1.0 / d_in
                                else:
                                    weight = 0.1 # Fallback
                            except:
                                # Fallback nếu graph không hỗ trợ in_degree chuẩn
                                weight = 0.1

                            # Tung xúc xắc
                            if random.random() < weight:
                                next_newly_active.add(neighbor)
            
            # 2. Cập nhật trạng thái
            unique_new = next_newly_active - active_nodes
            if not unique_new:
                break 
                
            active_nodes.update(unique_new)
            newly_active = unique_new
            
        total_spreads.append(len(active_nodes))

    # Tính toán thống kê
    mean_spread = np.mean(total_spreads)
    std_spread = np.std(total_spreads)
    
    return mean_spread, std_spread

# ==========================================
# 2. QUY TRÌNH BENCHMARK TỔNG HỢP
# ==========================================
def run_benchmark():
    # Load Config
    config_path = "configs/hyperparams.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Load Graph Data
    graph_path = os.path.join(cfg['dataset']['processed_dir'], "multiplex_graph.pkl")
    with open(graph_path, 'rb') as f:
        graph_data = pickle.load(f)
    
    # === CẤU HÌNH BENCHMARK ===
    budgets = [10, 20, 30, 40, 50]  
    n_mc_runs = 200                 
    results = []

    logger.info(f"📊 BẮT ĐẦU BENCHMARKING (Budgets: {budgets})")
    logger.info(f"⚙️ Simulation Model: Weighted Cascade (p = 1/d_in)")

    for k in budgets:
        logger.info(f"\n--- Running for Budget K={k} ---")
        
        # Bước 1: Tìm seeds bằng REM
        seeds = robust_inference(budget_k=k, num_restarts=5, steps=100) 
        
        # Bước 2: Verify bằng Monte Carlo chuẩn
        mean_spread, std_spread = run_stable_monte_carlo(seeds, graph_data, n_simulations=n_mc_runs)
        
        logger.info(f"✅ K={k} | Real Spread: {mean_spread:.2f} ± {std_spread:.2f}")
        
        results.append({
            "Budget": k,
            "Mean_Spread": mean_spread,
            "Std_Dev": std_spread,
            "Seeds": str(seeds)
        })

    # ==========================================
    # 3. XUẤT DỮ LIỆU & VẼ BIỂU ĐỒ
    # ==========================================
    df = pd.DataFrame(results)
    
    # Lưu CSV
    csv_filename = "benchmark_results_rem.csv"
    df.to_csv(csv_filename, index=False)
    logger.info(f"💾 Đã lưu dữ liệu: {csv_filename}")

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    plt.plot(df["Budget"], df["Mean_Spread"], marker='o', linewidth=2, color='b', label='REM (Ours)')
    plt.fill_between(df["Budget"], 
                     df["Mean_Spread"] - df["Std_Dev"], 
                     df["Mean_Spread"] + df["Std_Dev"], 
                     color='b', alpha=0.1, label='Std Dev')

    plt.title("Influence Maximization Performance (REM - Weighted Cascade)", fontsize=14)
    plt.xlabel("Seed Set Size (Budget k)", fontsize=12)
    plt.ylabel("Influence Spread (Expected Nodes)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    img_filename = "rem_performance_chart.png"
    plt.savefig(img_filename, dpi=300)
    logger.info(f"📈 Đã lưu biểu đồ: {img_filename}")
    # plt.show() # Tắt show nếu chạy trên server không có màn hình

    print("\n" + "="*40)
    print("KẾT QUẢ TỔNG HỢP:")
    print(df[["Budget", "Mean_Spread", "Std_Dev"]])
    print("="*40)

if __name__ == "__main__":
    run_benchmark()