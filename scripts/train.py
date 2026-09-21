import sys
import os
import time
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Data, Batch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
from src.data.dataset import REMDataset, load_graph
from src.models.seed2vec import Seed2Vec
from src.models.pmoe import PMoE

log = logging.getLogger("REM")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def train(config_path="configs/hyperparams.yaml", override_epochs=None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = cfg["dataset"].get("name", "dataset")
    log.info(f"Training [{dataset_name}] on device: {device}")

    num_nodes, edge_index, _ = load_graph(cfg, device)
    log.info(f"Graph: {num_nodes} nodes, {edge_index.size(1)} unified edges")

    data_path = os.path.join(cfg["dataset"]["processed_dir"], "augmented_data.SG")
    if not os.path.exists(data_path):
        data_path = os.path.join(cfg["dataset"]["processed_dir"], "train_data.SG")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No training data found in {cfg['dataset']['processed_dir']}. Run preprocess.py first.")

    dataset = REMDataset(data_path)
    total_len = len(dataset)
    val_len = max(1, int(total_len * 0.2)) if total_len >= 5 else 0
    train_len = total_len - val_len

    if val_len > 0:
        train_set, val_set = random_split(
            dataset, [train_len, val_len],
            generator=torch.Generator().manual_seed(cfg["project"].get("seed", 42))
        )
        train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=cfg["train"]["batch_size"], shuffle=False)
        log.info(f"Dataset split: {train_len} Train samples | {val_len} Validation samples (80/20)")
    else:
        train_loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=True)
        val_loader = None
        log.info(f"Dataset: {total_len} samples (no validation split)")

    vae = Seed2Vec(num_nodes, cfg["model"]["latent_dim"], cfg["model"]["hidden_dim"]).to(device)
    pmoe = PMoE(num_nodes, cfg["model"]["num_experts"], hidden_dim=cfg["model"]["hidden_dim"]).to(device)

    opt_vae = optim.Adam(vae.parameters(), lr=cfg["train"]["lr_vae"])
    opt_pmoe = optim.Adam(pmoe.parameters(), lr=cfg["train"]["lr_pmoe"])
    kl_w = cfg["train"]["kl_weight"]
    epochs = override_epochs if override_epochs is not None else cfg["train"]["epochs"]

    v_ckpt = "checkpoints/seed2vec.pth" if dataset_name == "Celegans" else f"checkpoints/seed2vec_{dataset_name}.pth"
    p_ckpt = "checkpoints/pmoe.pth" if dataset_name == "Celegans" else f"checkpoints/pmoe_{dataset_name}.pth"
    os.makedirs("checkpoints", exist_ok=True)

    best_val_loss = float("inf")

    log.info(f"\n{'='*75}\nSTARTING TRAINING: {epochs} EPOCHS | Hidden={cfg['model']['hidden_dim']} | Experts={cfg['model']['num_experts']}\n{'='*75}")

    for epoch in range(epochs):
        t0 = time.time()
        vae.train(); pmoe.train()
        train_vae_loss = train_pmoe_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # VAE step
            opt_vae.zero_grad()
            recon, mu, logvar = vae(x)
            mse = torch.nn.functional.mse_loss(recon, x, reduction="sum")
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            vae_loss = mse + kl_w * kld
            vae_loss.backward()
            opt_vae.step()

            # PMoE step
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

            train_vae_loss += vae_loss.item()
            train_pmoe_loss += pmoe_loss.item()

        train_vae_avg = train_vae_loss / train_len if train_len > 0 else 0.0
        train_pmoe_avg = train_pmoe_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        # Validation phase
        val_vae_avg = val_pmoe_avg = 0.0
        if val_loader is not None:
            vae.eval(); pmoe.eval()
            val_vae_loss = val_pmoe_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    recon, mu, logvar = vae(x)
                    mse = torch.nn.functional.mse_loss(recon, x, reduction="sum")
                    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    val_vae_loss += (mse + kl_w * kld).item()

                    batch_data = Batch.from_data_list([
                        Data(x=x[i].unsqueeze(-1), edge_index=edge_index)
                        for i in range(x.size(0))
                    ]).to(device)
                    pred = pmoe(batch_data).sum(dim=1).squeeze()
                    target = y.sum(dim=1) if y.dim() > 1 else y
                    val_pmoe_loss += torch.nn.functional.mse_loss(pred, target.float()).item()

            val_vae_avg = val_vae_loss / val_len
            val_pmoe_avg = val_pmoe_loss / len(val_loader)

        elapsed = time.time() - t0
        total_val_metric = val_pmoe_avg if val_loader is not None else train_pmoe_avg

        is_best = total_val_metric < best_val_loss
        status_msg = ""
        if is_best:
            best_val_loss = total_val_metric
            torch.save(vae.state_dict(), v_ckpt)
            torch.save(pmoe.state_dict(), p_ckpt)
            status_msg = "--> [BEST MODEL SAVED]"

        if val_loader is not None:
            log.info(
                f"Epoch [{epoch+1:02d}/{epochs:02d}] "
                f"| Tr VAE: {train_vae_avg:7.2f} | Val VAE: {val_vae_avg:7.2f} "
                f"| Tr PMoE: {train_pmoe_avg:8.2f} | Val PMoE: {val_pmoe_avg:8.2f} "
                f"| Time: {elapsed:4.1f}s {status_msg}"
            )
        else:
            log.info(
                f"Epoch [{epoch+1:02d}/{epochs:02d}] "
                f"| VAE Loss: {train_vae_avg:7.2f} | PMoE Loss: {train_pmoe_avg:8.2f} "
                f"| Time: {elapsed:4.1f}s {status_msg}"
            )

    log.info(f"\n{'='*75}\nTRAINING COMPLETED: Best Model Checkpoint at {v_ckpt}\n{'='*75}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hyperparams.yaml", help="Path to config YAML")
    ap.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    args = ap.parse_args()
    train(config_path=args.config, override_epochs=args.epochs)