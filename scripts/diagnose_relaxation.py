"""diagnose_relaxation.py — Rigorous Diagnostic of Relaxation Gap & Optimization Trajectory.

Conducts two peer-review diagnostic experiments:
1. Relaxation Gap Verification:
   Compares PMoE predictive fidelity on continuous soft activations x_soft in [0, 1]^N
   versus discrete projected binary vectors x_hard in {0, 1}^N against true MC spread.
2. Latent Optimization Trajectory Trace:
   Logs step-by-step evolution of [sum(x), PMoE(x_soft), PMoE(x_hard), true_MC(x_hard)]
   to detect surrogate exploitation and objective conflict.

Usage:
    python scripts/diagnose_relaxation.py --config configs/hyperparams.yaml --budget 20
"""
import os
import sys
import yaml
import argparse
import numpy as np
import torch
import torch.optim as optim
from scipy import stats
from torch_geometric.data import Data, Batch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import load_graph
from src.data.simulation import run_IC_simulation
from src.models.seed2vec import Seed2Vec
from src.models.pmoe import PMoE


def run_experiment_1_relaxation_gap(vae, pmoe, graph_list, edge_index, num_nodes, device, budget_k=20, num_samples=25, mc_runs=15):
    """Experiment 1: Evaluates whether PMoE degrades when evaluated on continuous/soft x vs binary x."""
    print("\n" + "=" * 78)
    print(f" EXPERIMENT 1: RELAXATION GAP ANALYSIS (N={num_samples} latent samples | k={budget_k})")
    print("=" * 78)

    pmoe_soft_list = []
    pmoe_hard_list = []
    true_mc_list = []
    sum_soft_list = []

    pmoe.eval()
    vae.eval()

    with torch.no_grad():
        for i in range(num_samples):
            torch.manual_seed(i * 50 + 7)
            z = torch.randn(1, vae.fc_mu.out_features, device=device)
            x_soft = vae.decode(z).squeeze()  # [num_nodes]
            sum_soft = x_soft.sum().item()
            sum_soft_list.append(sum_soft)

            # 1. PMoE on continuous soft vector
            data_soft = Batch.from_data_list([Data(x=x_soft.view(-1, 1), edge_index=edge_index)]).to(device)
            pmoe_soft = pmoe(data_soft).sum().item()
            pmoe_soft_list.append(pmoe_soft)

            # 2. Project to discrete hard top-k
            _, top_idx = torch.topk(x_soft, k=budget_k)
            seeds = top_idx.cpu().numpy().tolist()

            x_hard = torch.zeros(num_nodes, 1, device=device)
            x_hard[seeds] = 1.0
            data_hard = Batch.from_data_list([Data(x=x_hard, edge_index=edge_index)]).to(device)
            pmoe_hard = pmoe(data_hard).sum().item()
            pmoe_hard_list.append(pmoe_hard)

            # 3. Ground-truth Monte Carlo on seeds
            mc_spread = np.mean([len(run_IC_simulation(graph_list, seeds)) for _ in range(mc_runs)])
            true_mc_list.append(mc_spread)

    pmoe_soft_list = np.array(pmoe_soft_list)
    pmoe_hard_list = np.array(pmoe_hard_list)
    true_mc_list = np.array(true_mc_list)
    sum_soft_list = np.array(sum_soft_list)

    # Correlations with Ground Truth MC
    r_soft, p_soft = stats.pearsonr(pmoe_soft_list, true_mc_list)
    rho_soft, p_rho_soft = stats.spearmanr(pmoe_soft_list, true_mc_list)

    r_hard, p_hard = stats.pearsonr(pmoe_hard_list, true_mc_list)
    rho_hard, p_rho_hard = stats.spearmanr(pmoe_hard_list, true_mc_list)

    print(f"Mean sum(x_soft) across random z    : {np.mean(sum_soft_list):.2f} (Target k = {budget_k})")
    print(f"Mean PMoE(x_soft)                    : {np.mean(pmoe_soft_list):.2f}")
    print(f"Mean PMoE(x_hard)                    : {np.mean(pmoe_hard_list):.2f}")
    print(f"Mean True MC Spread(seeds)           : {np.mean(true_mc_list):.2f}")
    print("-" * 78)
    print(f"PMoE Soft vs True MC : Pearson r = {r_soft:.4f} (p={p_soft:.2e}) | Spearman rho = {rho_soft:.4f} (p={p_rho_soft:.2e})")
    print(f"PMoE Hard vs True MC : Pearson r = {r_hard:.4f} (p={p_hard:.2e}) | Spearman rho = {rho_hard:.4f} (p={p_rho_hard:.2e})")
    print("-" * 78)

    gap = rho_hard - rho_soft
    if gap > 0.15:
        verdict = "SEVERE RELAXATION GAP: PMoE ranking degrades significantly on continuous activations!"
    elif gap > 0.05:
        verdict = "MODERATE RELAXATION GAP: Continuous activations introduce measurable surrogate noise."
    else:
        verdict = "MINIMAL RELAXATION GAP: Continuous surrogate tracks binary ranking reasonably well."
    print(f"Diagnostic Assessment: {verdict}")
    print("=" * 78 + "\n")


def run_experiment_2_trajectory_trace(vae, pmoe, graph_list, edge_index, num_nodes, device, budget_k=20, steps=50, mc_runs=15):
    """Experiment 2: Step-by-step trace of latent gradient ascent to visualize the objective landscape."""
    print("\n" + "=" * 85)
    print(f" EXPERIMENT 2: LATENT OPTIMIZATION TRAJECTORY TRACE (k={budget_k} | T={steps} steps)")
    print("=" * 85)

    torch.manual_seed(42)
    z = torch.randn(1, vae.fc_mu.out_features, device=device, requires_grad=True)
    opt = optim.Adam([z], lr=0.2)
    lam = 0.5

    rows = []
    print(f"{'Step':<5} | {'sum(x_hat)':<10} | {'PMoE(x_soft)':<12} | {'PMoE(x_hard)':<12} | {'True MC':<8} | {'J(S_t, S_0)':<10} | {'J(S_t, S_prev)':<12} | {'Status'}")
    print("-" * 92)

    s0_set = None
    sprev_set = None

    for t in range(steps + 1):
        x_hat = vae.decode(z)
        data_soft = Batch.from_data_list([Data(x=x_hat.view(-1, 1), edge_index=edge_index)]).to(device)
        spread_soft = pmoe(data_soft).sum()
        sum_x = x_hat.sum().item()

        # Hard projection
        with torch.no_grad():
            _, idx = torch.topk(x_hat.squeeze(), k=budget_k)
            seeds = idx.cpu().numpy().tolist()
            st_set = set(seeds)

            if t == 0:
                s0_set = set(seeds)
                sprev_set = set(seeds)

            # Jaccard similarities
            j_0 = len(st_set & s0_set) / len(st_set | s0_set) if len(st_set | s0_set) > 0 else 1.0
            j_prev = len(st_set & sprev_set) / len(st_set | sprev_set) if len(st_set | sprev_set) > 0 else 1.0
            sprev_set = st_set

            x_hard = torch.zeros(num_nodes, 1, device=device)
            x_hard[seeds] = 1.0
            data_hard = Batch.from_data_list([Data(x=x_hard, edge_index=edge_index)]).to(device)
            spread_hard = pmoe(data_hard).sum().item()

        # Evaluate MC at intervals
        if t % 5 == 0 or t == steps:
            mc_spread = np.mean([len(run_IC_simulation(graph_list, seeds)) for _ in range(mc_runs)])
            status = "Start" if t == 0 else ("Final" if t == steps else "Tracking")
            print(f"{t:<5d} | {sum_x:10.2f} | {spread_soft.item():12.2f} | {spread_hard:12.2f} | {mc_spread:8.2f} | {j_0:10.3f} | {j_prev:12.3f} | {status}")
            rows.append((t, sum_x, spread_soft.item(), spread_hard, mc_spread, j_0, j_prev))

        # Perform gradient ascent step
        if t < steps:
            opt.zero_grad()
            budget_loss = ((x_hat.sum() - budget_k) / max(1, budget_k)) ** 2
            eps = 1e-7
            entropy = -torch.mean(x_hat * torch.log(x_hat + eps) + (1 - x_hat) * torch.log(1 - x_hat + eps))
            loss = -spread_soft + lam * budget_k * budget_loss + 0.1 * entropy
            loss.backward()
            opt.step()

    print("-" * 92)
    init_mc = rows[0][4]
    final_mc = rows[-1][4]
    gain = ((final_mc - init_mc) / init_mc * 100.0) if init_mc > 0 else 0.0
    print(f"Initial True MC Spread: {init_mc:.2f}  --->  Final True MC Spread: {final_mc:.2f} (Gain: {gain:+.1f}%)")
    print(f"Initial sum(x_hat)     : {rows[0][1]:.2f}  --->  Final sum(x_hat)     : {rows[-1][1]:.2f} (Target: {budget_k})")
    print(f"Final Top-K Jaccard Stability: J(S_final, S_0) = {rows[-1][5]:.3f} | Mean Step Stability J(S_t, S_prev) = {np.mean([r[6] for r in rows[1:]]):.3f}")
    print("=" * 92 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hyperparams.yaml")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--mc-runs", type=int, default=12)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_nodes, edge_index, graph_list = load_graph(cfg, device)
    dataset_name = cfg["dataset"].get("name", "Celegans")

    vae = Seed2Vec(num_nodes, cfg["model"]["latent_dim"], cfg["model"]["hidden_dim"]).to(device)
    pmoe = PMoE(num_nodes, cfg["model"]["num_experts"], hidden_dim=cfg["model"]["hidden_dim"]).to(device)

    v_ckpt = f"checkpoints/seed2vec_{dataset_name}.pth" if os.path.exists(f"checkpoints/seed2vec_{dataset_name}.pth") else "checkpoints/seed2vec.pth"
    p_ckpt = f"checkpoints/pmoe_{dataset_name}.pth" if os.path.exists(f"checkpoints/pmoe_{dataset_name}.pth") else "checkpoints/pmoe.pth"

    if os.path.exists(v_ckpt):
        vae.load_state_dict(torch.load(v_ckpt, map_location=device))
    if os.path.exists(p_ckpt):
        pmoe.load_state_dict(torch.load(p_ckpt, map_location=device))

    # Run Diagnostic 1
    run_experiment_1_relaxation_gap(vae, pmoe, graph_list, edge_index, num_nodes, device,
                                   budget_k=args.budget, num_samples=args.num_samples, mc_runs=args.mc_runs)

    # Run Diagnostic 2
    run_experiment_2_trajectory_trace(vae, pmoe, graph_list, edge_index, num_nodes, device,
                                     budget_k=args.budget, steps=50, mc_runs=args.mc_runs)


if __name__ == "__main__":
    main()
