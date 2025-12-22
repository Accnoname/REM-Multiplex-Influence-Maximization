import sys
import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_geometric.data import Data, Batch # <--- IMPORT MỚI

# Thêm thư mục gốc vào path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset_loader import REMDataset
from src.models.seed2vec import Seed2Vec
from src.models.pmoe import PMoE
from src.utils.logger import setup_logger

logger = setup_logger() 

def train():
    # 1. Load Config
    config_path = "configs/hyperparams.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['project']['device'] if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {device}")

    # 2. Load Data & Graph Structure
    data_path = os.path.join(cfg['dataset']['processed_dir'], "augmented_data.SG")
    graph_path = os.path.join(cfg['dataset']['processed_dir'], "multiplex_graph.pkl")
    
    import pickle
    with open(graph_path, 'rb') as f:
        graph_data = pickle.load(f)
    
    from torch_geometric.utils import from_networkx
    
    # Chuẩn bị cấu trúc đồ thị (Edge Index) dùng chung cho cả batch
    # Giả sử dùng layer 1 (index 1) làm cấu trúc lan truyền chính
    first_layer_graph = graph_data['graphs'][1] 
    pyg_graph = from_networkx(first_layer_graph)
    main_edge_index = pyg_graph.edge_index.to(device)

    dataset = REMDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=cfg['train']['batch_size'], shuffle=True)
    
    # 3. Initialize Models
    num_nodes = graph_data['num_nodes']
    
    vae = Seed2Vec(
        num_nodes=num_nodes,
        latent_dim=cfg['model']['latent_dim'],
        hidden_dim=cfg['model']['hidden_dim'],
        dropout=cfg['model']['dropout']
    ).to(device)
    
    pmoe = PMoE(
        num_nodes=num_nodes,
        num_experts=cfg['model']['num_experts'],
        hidden_dim=cfg['model']['hidden_dim']
    ).to(device)

    # 4. Optimizers
    opt_vae = optim.Adam(vae.parameters(), lr=cfg['train']['lr_vae'])
    opt_pmoe = optim.Adam(pmoe.parameters(), lr=cfg['train']['lr_pmoe'])

    # 5. Training Loop
    epochs = cfg['train']['epochs']
    kl_weight = cfg['train']['kl_weight']
    
    logger.info("Start Training...")
    
    for epoch in range(epochs):
        vae.train()
        pmoe.train()
        
        total_vae_loss = 0
        total_pmoe_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_x, batch_y in progress_bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # --- Train Seed2Vec (VAE) ---
            opt_vae.zero_grad()
            recon_x, mu, logvar = vae(batch_x)
            
            mse_loss = torch.nn.functional.mse_loss(recon_x, batch_x, reduction='sum')
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            vae_loss = mse_loss + kl_weight * kld_loss
            
            vae_loss.backward()
            opt_vae.step()
            
            # --- Train PMoE ---
            # [ENGINEER FIX]: Chuyển đổi batch 3D sang PyG Batch Object (2D disjoint union)
            # để tránh lỗi "Static graphs not supported" của GATConv
            
            data_list = []
            curr_batch_size = batch_x.size(0)
            
            for i in range(curr_batch_size):
                # Lấy vector features của graph thứ i
                x_i = batch_x[i].float() # Đảm bảo là float
                
                # [QUAN TRỌNG]: Nếu x_i là 1D [Num_Nodes], phải unsqueeze thành [Num_Nodes, 1]
                if x_i.dim() == 1:
                    x_i = x_i.unsqueeze(-1)
                
                # Tạo Data object.
                # Lưu ý: edge_index dùng chung, không cần .clone() nếu chỉ đọc
                data = Data(x=x_i, edge_index=main_edge_index)
                data_list.append(data)
            
            # Tạo Batch lớn và đưa lên device
            batch_input = Batch.from_data_list(data_list).to(device)
            
            opt_pmoe.zero_grad()
            
            # PMoE forward với Batch object
            # Output pred_y shape: [Batch_Size, Num_Nodes, 1]
            pred_y_map = pmoe(batch_input) 
            
            # Tính tổng influence spread dự đoán (Sum over nodes)
            # Output: [Batch_Size]
            pred_spread = pred_y_map.sum(dim=1).squeeze()
            
            # --- [ENGINEER FIX] ---
            # batch_y đang là [Batch, Nodes] (chi tiết từng node), cần sum lại thành [Batch] (tổng số node)
            # để khớp với pred_spread (tổng độ lan truyền dự đoán).
            if batch_y.dim() > 1:
                target_spread = batch_y.sum(dim=1)
            else:
                target_spread = batch_y
            
            # Đảm bảo shape khớp nhau 100% trước khi đưa vào Loss
            # Nếu pred_spread là [32], target_spread cũng phải là [32]
            pmoe_loss = torch.nn.functional.mse_loss(pred_spread, target_spread.float())
            
            pmoe_loss.backward()
            opt_pmoe.step()
            
            total_vae_loss += vae_loss.item()
            total_pmoe_loss += pmoe_loss.item()
            
            progress_bar.set_postfix({
                "VAE": f"{vae_loss.item()/len(batch_x):.2f}", 
                "PMoE": f"{pmoe_loss.item():.2f}"
            })

        avg_vae = total_vae_loss / len(dataset)
        avg_pmoe = total_pmoe_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} Summary | VAE Loss: {avg_vae:.4f} | PMoE Loss: {avg_pmoe:.4f}")

    # 6. Save Models
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(vae.state_dict(), "checkpoints/seed2vec.pth")
    torch.save(pmoe.state_dict(), "checkpoints/pmoe.pth")
    logger.info("Training Finished! Models saved to 'checkpoints/'")

if __name__ == "__main__":
    train()