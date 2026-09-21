import sys
import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
from src.data.dataset import REMDataset, load_graph
from src.models.seed2vec import Seed2Vec
from src.models.pmoe import PMoE

log = logging.getLogger("REM")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def train():
    with open("configs/hyperparams.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    num_nodes, edge_index, _ = load_graph(cfg, device)
    log.info(f"Graph: {num_nodes} nodes, {edge_index.size(1)} unified edges")

    data_path = os.path.join(cfg["dataset"]["processed_dir"], "augmented_data.SG")
    dataset = REMDataset(data_path)
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=True)

    vae = Seed2Vec(num_nodes, cfg["model"]["latent_dim"], cfg["model"]["hidden_dim"]).to(device)
    pmoe = PMoE(num_nodes, cfg["model"]["num_experts"], hidden_dim=cfg["model"]["hidden_dim"]).to(device)

    opt_vae = optim.Adam(vae.parameters(), lr=cfg["train"]["lr_vae"])
    opt_pmoe = optim.Adam(pmoe.parameters(), lr=cfg["train"]["lr_pmoe"])
    kl_w = cfg["train"]["kl_weight"]
    epochs = cfg["train"]["epochs"]

    for epoch in range(epochs):
        vae.train(); pmoe.train()
        total_vae = total_pmoe = 0.0

        for x, y in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)

            # VAE step
            opt_vae.zero_grad()
            recon, mu, logvar = vae(x)
            mse = torch.nn.functional.mse_loss(recon, x, reduction="sum")
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            vae_loss = mse + kl_w * kld
            vae_loss.backward()
            opt_vae.step()

            # PMoE step — build PyG batch from current x
            opt_pmoe.zero_grad()
            batch_data = Batch.from_data_list([
                Data(x=x[i].unsqueeze(-1), edge_index=edge_index)
                for i in range(x.size(0))
            ]).to(device)
            pred = pmoe(batch_data).sum(dim=1).squeeze()
            target = y.sum(dim=1) if y.dim() > 1 else y
            pmoe_loss = torch.nn.functional.mse_loss(pred, target.float())
            pmoe_loss.backward()
            opt_pmoe.step()

            total_vae += vae_loss.item()
            total_pmoe += pmoe_loss.item()

        log.info(f"Epoch {epoch+1} | VAE: {total_vae/len(dataset):.4f} | PMoE: {total_pmoe/len(loader):.4f}")

        if (epoch + 1) % 2 == 0 or (epoch + 1) == epochs:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(vae.state_dict(), "checkpoints/seed2vec.pth")
            torch.save(pmoe.state_dict(), "checkpoints/pmoe.pth")
            log.info(f"Checkpoint saved (epoch {epoch+1})")

    log.info("Training done.")


if __name__ == "__main__":
    train()