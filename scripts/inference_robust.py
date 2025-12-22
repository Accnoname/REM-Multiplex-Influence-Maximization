import sys
import os
import yaml
import torch
import torch.optim as optim
from torch_geometric.data import Data, Batch
import numpy as np

# Giả định import các module của bạn
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.seed2vec import Seed2Vec
from src.models.pmoe import PMoE
from src.utils.logger import setup_logger

logger = setup_logger()

def load_graph_structure(cfg, device):
    import pickle
    from torch_geometric.utils import from_networkx
    graph_path = os.path.join(cfg['dataset']['processed_dir'], "multiplex_graph.pkl")
    with open(graph_path, 'rb') as f:
        graph_data = pickle.load(f)
    # Lấy cấu trúc cạnh để PMoE tính toán
    first_layer = graph_data['graphs'][1]
    pyg_graph = from_networkx(first_layer)
    return graph_data['num_nodes'], pyg_graph.edge_index.to(device)

def robust_inference(budget_k=50, num_restarts=10, steps=150):
    """
    Thực hiện Algorithm 2: Tìm kiếm hạt giống tối ưu trong không gian ẩn.
    """
    # 1. Load Config & Device
    config_path = "configs/hyperparams.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg['project']['device'] if torch.cuda.is_available() else "cpu")
    num_nodes, edge_index = load_graph_structure(cfg, device)
    
    # 2. Init & Load Models (Chế độ Eval cực quan trọng)
    vae = Seed2Vec(num_nodes, cfg['model']['latent_dim'], cfg['model']['hidden_dim']).to(device)
    pmoe = PMoE(num_nodes, cfg['model']['num_experts'], in_channels=1, hidden_dim=cfg['model']['hidden_dim']).to(device)
    
    # Load weights (Map location để tránh lỗi device)
    vae.load_state_dict(torch.load("checkpoints/seed2vec.pth", map_location=device))
    pmoe.load_state_dict(torch.load("checkpoints/pmoe.pth", map_location=device))
    vae.eval()
    pmoe.eval()

    best_global_spread = -float('inf')
    best_global_seeds = []
    
    # [DEBUG] Biến để lưu tập seed của vòng lặp trước đó
    prev_seeds = None

    logger.info(f"🚀 BẮT ĐẦU CHIẾN DỊCH: Robust Inference ({num_restarts} restarts)")

    for i in range(num_restarts):
        # [QUAN TRỌNG] Thay đổi seed ngẫu nhiên
        torch.manual_seed(i * 100) 
        
        # Khởi tạo z ngẫu nhiên
        z = torch.randn(1, cfg['model']['latent_dim'], device=device, requires_grad=True)
        
        # [DEBUG LOG] In ra 5 giá trị đầu tiên của z lúc khởi tạo
        start_z_sample = z[0, :5].detach().cpu().numpy()
        logger.info(f"Run {i+1:02d} | Start z (first 5): {start_z_sample}")

        # Optimizer riêng cho z
        optimizer = optim.Adam([z], lr=0.5) 
        
        current_run_best_spread = 0
        
        # 4. Gradient Ascent Loop (Leo đồi)
        for step in range(steps):
            optimizer.zero_grad()
            
            x_hat = vae.decode(z)
            
            # Chuẩn bị data cho PMoE
            x_input = x_hat.view(-1, 1) 
            data = Data(x=x_input, edge_index=edge_index)
            batch = Batch.from_data_list([data]).to(device)
            
            pred_spread = pmoe(batch).sum()
            
            loss = -pred_spread
            loss.backward()
            optimizer.step()
            
            if pred_spread.item() > current_run_best_spread:
                current_run_best_spread = pred_spread.item()

        # [DEBUG LOG] In ra 5 giá trị đầu tiên của z SAU KHI tối ưu xong
        end_z_sample = z[0, :5].detach().cpu().numpy()
        
        # 5. Giải mã (Decoding)
        with torch.no_grad():
            final_reconstructed = vae.decode(z).squeeze()
            val, indices = torch.topk(final_reconstructed, k=budget_k)
            current_seeds = indices.cpu().numpy().tolist()
        
        # [DEBUG LOG] So sánh với lần chạy trước
        diff_msg = ""
        if prev_seeds is not None:
            # Tìm sự khác biệt giữa tập seed hiện tại và tập trước
            diff_set = set(current_seeds) - set(prev_seeds)
            if len(diff_set) == 0:
                diff_msg = "⚠️ GIỐNG HỆT lần trước (Identical)"
            else:
                diff_msg = f"✅ KHÁC BIỆT {len(diff_set)} nodes"
        
        # Cập nhật prev_seeds
        prev_seeds = current_seeds

        logger.info(f"➤ Run {i+1:02d}: Pred = {current_run_best_spread:.2f} | End z: {end_z_sample} | {diff_msg}")

        # Cập nhật kết quả tốt nhất toàn cục
        if current_run_best_spread > best_global_spread:
            best_global_spread = current_run_best_spread
            best_global_seeds = current_seeds
    
    return best_global_seeds

if __name__ == "__main__":
    # Chạy inference
    seeds = robust_inference(budget_k=50, num_restarts=10)
    
    # Gợi ý bước tiếp theo
    print("\n[HƯỚNG DẪN]: Copy list 'seeds' trên vào file 'verify_multiplex.py' để chạy Monte Carlo kiểm chứng kết quả thực tế (dự kiến ~463).")