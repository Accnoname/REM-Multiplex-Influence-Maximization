import sys
import os
import yaml
import random
import torch
import torch.optim as optim
from torch_geometric.data import Data, Batch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
from src.data.dataset import load_graph
from src.models.seed2vec import Seed2Vec
from src.models.pmoe import PMoE

log = logging.getLogger("REM")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _multiplex_degree(graph_list):
    """Sum degree across all layers per node."""
    deg = {}
    for G in graph_list:
        for n in G.nodes():
            deg[n] = deg.get(n, 0) + G.degree(n)
    return deg


def local_search(seeds, graph_list, max_swaps=3):
    """Swap lowest-degree seeds for higher-degree neighbors (1-hop)."""
    deg = _multiplex_degree(graph_list)
    seed_set = set(seeds)

    candidates = {
        nbr
        for s in seeds
        for G in graph_list if G.has_node(s)
        for nbr in G.neighbors(s)
        if nbr not in seed_set
    }
    if not candidates:
        return seeds

    best_cands = sorted(candidates, key=lambda c: deg.get(c, 0), reverse=True)
    result = list(seeds)
    swaps = 0
    for weak in sorted(result, key=lambda s: deg.get(s, 0)):
        if swaps >= max_swaps or not best_cands:
            break
        cand = best_cands.pop(0)
        if deg.get(cand, 0) > deg.get(weak, 0) * 1.5:
            result.remove(weak)
            result.append(cand)
            seed_set.discard(weak)
            seed_set.add(cand)
            swaps += 1
    return result


def robust_inference(budget_k=50, num_restarts=3, steps=60, config_path="configs/hyperparams.yaml"):
    """REM Algorithm 2: latent-space gradient ascent + local search."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_nodes, edge_index, graph_list = load_graph(cfg, device)

    vae = Seed2Vec(num_nodes, cfg["model"]["latent_dim"], cfg["model"]["hidden_dim"]).to(device)
    pmoe = PMoE(num_nodes, cfg["model"]["num_experts"], hidden_dim=cfg["model"]["hidden_dim"]).to(device)

    for ckpt, model, name in [
        ("checkpoints/seed2vec.pth", vae, "Seed2Vec"),
        ("checkpoints/pmoe.pth", pmoe, "PMoE"),
    ]:
        if os.path.exists(ckpt):
            try:
                sd = torch.load(ckpt, map_location=device)
                # Remap legacy key names (enc1/dec1 → enc.0/dec.0) if needed
                remap = {
                    "enc1.weight": "enc.0.weight", "enc1.bias": "enc.0.bias",
                    "enc2.weight": "enc.3.weight", "enc2.bias": "enc.3.bias",
                    "dec1.weight": "dec.0.weight", "dec1.bias": "dec.0.bias",
                    "dec2.weight": "dec.3.weight", "dec2.bias": "dec.3.bias",
                    "dec_out.weight": "dec.5.weight", "dec_out.bias": "dec.5.bias",
                }
                sd = {remap.get(k, k): v for k, v in sd.items()}
                model.load_state_dict(sd)
                log.info(f"Loaded {name} checkpoint")
            except Exception as e:
                log.warning(f"{name} checkpoint mismatch: {e}")


    vae.eval(); pmoe.eval()

    inf_cfg = cfg.get("inference", {})
    lr_z = inf_cfg.get("lr_z", 0.2)
    lam = inf_cfg.get("budget_penalty", 0.5)

    log.info(f"Inference | k={budget_k} | restarts={num_restarts} | steps={steps}")

    best_score, best_seeds, prev_seeds = -float("inf"), [], None

    for i in range(num_restarts):
        torch.manual_seed(i * 100 + budget_k)
        random.seed(i * 100 + budget_k)

        z = torch.randn(1, cfg["model"]["latent_dim"], device=device, requires_grad=True)
        opt = optim.Adam([z], lr=lr_z)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.01)

        run_best_score, run_best_x = -float("inf"), None

        for _ in range(steps):
            opt.zero_grad()
            x_hat = vae.decode(z)
            data = Batch.from_data_list([Data(x=x_hat.view(-1, 1), edge_index=edge_index)]).to(device)
            spread = pmoe(data).sum()

            budget_loss = ((x_hat.sum() - budget_k) / max(1, budget_k)) ** 2
            eps = 1e-7
            entropy = -torch.mean(
                x_hat * torch.log(x_hat + eps) + (1 - x_hat) * torch.log(1 - x_hat + eps)
            )
            (-spread + lam * budget_k * budget_loss + 0.1 * entropy).backward()
            opt.step(); sched.step()

            score = spread.item() - budget_loss.item()
            if score > run_best_score:
                run_best_score, run_best_x = score, x_hat.detach().clone()

        with torch.no_grad():
            _, idx = torch.topk(run_best_x.squeeze(), k=budget_k)
            seeds = local_search(idx.cpu().numpy().tolist(), graph_list)

        diff = f"Changed {len(set(seeds)-set(prev_seeds))} nodes" if prev_seeds else ""
        log.info(f"Run {i+1:02d}: score={run_best_score:.2f} {diff}")
        prev_seeds = seeds

        if run_best_score > best_score:
            best_score, best_seeds = run_best_score, seeds

    log.info(f"Done k={budget_k} | best={best_score:.2f}")
    return best_seeds


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hyperparams.yaml", help="Path to config file")
    ap.add_argument("--budget", type=int, default=50, help="Seed budget k")
    ap.add_argument("--restarts", type=int, default=3, help="Number of random restarts")
    ap.add_argument("--steps", type=int, default=60, help="Gradient ascent steps")
    args = ap.parse_args()

    seeds = robust_inference(budget_k=args.budget, num_restarts=args.restarts, steps=args.steps, config_path=args.config)
    print(f"Seeds ({len(seeds)}): {seeds}")

