"""evaluate_surrogate.py — Surrogate Calibration & Cascade Distribution Analysis.

Performs research-grade validation:
1. PMoE Ranking Calibration: Evaluates Pearson (r) and Spearman (rho) correlation
   between PMoE surrogate predictions and true Monte Carlo diffusion spread.
2. Cascade Size Distribution: Analyzes empirical diffusion histograms and quantiles
   to characterize cascade heterogeneity and variance.

Usage:
    python scripts/evaluate_surrogate.py --config configs/hyperparams.yaml --num-sets 50 --mc-sims 30
"""
import os
import sys
import yaml
import random
import argparse
import numpy as np
import torch
from scipy import stats
from torch_geometric.data import Data, Batch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
from src.data.dataset import load_graph
from src.data.simulation import run_IC_simulation
from src.models.pmoe import PMoE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("SurrogateEval")


def evaluate_ranking_correlation(pmoe, graph_list, edge_index, num_nodes, device, num_sets=50, mc_sims=30):
    """Compute Pearson and Spearman rank correlation between PMoE predicted spread and ground-truth MC spread."""
    log.info(f"Generating {num_sets} diverse seed sets across budgets [10, 20, 30, 40, 50]...")
    budgets = [10, 20, 30, 40, 50]
    pmoe_preds = []
    mc_spreads = []

    pmoe.eval()
    with torch.no_grad():
        for i in range(num_sets):
            k = budgets[i % len(budgets)]
            seeds = random.sample(range(num_nodes), k)

            # 1. PMoE surrogate prediction
            x = torch.zeros(num_nodes, 1, device=device)
            x[seeds] = 1.0
            data = Batch.from_data_list([Data(x=x, edge_index=edge_index)]).to(device)
            pred = pmoe(data).sum().item()
            pmoe_preds.append(pred)

            # 2. Independent Monte Carlo ground truth
            spread_samples = [len(run_IC_simulation(graph_list, seeds)) for _ in range(mc_sims)]
            true_mean = float(np.mean(spread_samples))
            mc_spreads.append(true_mean)

    pmoe_preds = np.array(pmoe_preds)
    mc_spreads = np.array(mc_spreads)

    pearson_r, p_val_r = stats.pearsonr(pmoe_preds, mc_spreads)
    spearman_rho, p_val_rho = stats.spearmanr(pmoe_preds, mc_spreads)

    print("\n" + "=" * 70)
    print("      PMoE SURROGATE CALIBRATION & RANKING CORRELATION")
    print("=" * 70)
    print(f"Sample Size (Seed Sets)        : {num_sets}")
    print(f"Monte Carlo Simulations/Set    : {mc_sims}")
    print(f"Pearson Correlation (r)        : {pearson_r:.4f} (p = {p_val_r:.2e})")
    print(f"Spearman Rank Correlation (rho): {spearman_rho:.4f} (p = {p_val_rho:.2e})")
    print("-" * 70)
    if spearman_rho >= 0.7:
        verdict = "STRONG RANKING FIDELITY (Surrogate reliably ranks candidate seeds)"
    elif spearman_rho >= 0.4:
        verdict = "MODERATE RANKING FIDELITY (Directionally usable, but noisy ranking)"
    else:
        verdict = "WEAK / UNRELIABLE RANKING (Susceptible to surrogate exploitation)"
    print(f"Diagnostic Assessment: {verdict}")
    print("=" * 70 + "\n")

    # 95% Confidence Interval for Pearson r via Fisher z-transform
    if len(pmoe_preds) > 3 and abs(pearson_r) < 1.0:
        z = np.arctanh(pearson_r)
        se_z = 1.0 / np.sqrt(len(pmoe_preds) - 3)
        ci_low = np.tanh(z - 1.96 * se_z)
        ci_high = np.tanh(z + 1.96 * se_z)
        ci_str = f"[{ci_low:.4f}, {ci_high:.4f}]"
    else:
        ci_str = "N/A"

    print("\n" + "=" * 75)
    print("      PMoE SURROGATE CALIBRATION & RANKING CORRELATION")
    print("=" * 75)
    print(f"Sample Size (Seed Sets)        : {num_sets}")
    print(f"Monte Carlo Simulations/Set    : {mc_sims}")
    print(f"Pearson Correlation (r)        : {pearson_r:.4f} (p = {p_val_r:.2e}) | 95% CI: {ci_str}")
    print(f"Spearman Rank Correlation (rho): {spearman_rho:.4f} (p = {p_val_rho:.2e})")
    print("-" * 75)
    if spearman_rho >= 0.7:
        verdict = "STRONG RANKING FIDELITY (Surrogate reliably preserves relative seed rankings)"
    elif spearman_rho >= 0.4:
        verdict = "MODERATE RANKING FIDELITY (Directionally usable, but noisy ranking)"
    else:
        verdict = "WEAK / UNRELIABLE RANKING (Susceptible to surrogate exploitation)"
    print(f"Diagnostic Assessment: {verdict}")
    print("=" * 75 + "\n")

    return pearson_r, spearman_rho


def print_ascii_histogram(data, label, bins=8):
    """Render an ASCII histogram for distribution inspection."""
    counts, edges = np.histogram(data, bins=bins)
    max_count = max(counts) if max(counts) > 0 else 1
    max_bar_width = 32

    print(f"\nEmpirical Cascade Distribution [{label}] (N={len(data)}):")
    print("-" * 58)
    for i in range(len(counts)):
        bar = "#" * int(counts[i] / max_count * max_bar_width)
        print(f"[{edges[i]:6.1f} - {edges[i+1]:6.1f}]: {bar:<33} ({counts[i]:3d})")
    print("-" * 58)


def analyze_cascade_heterogeneity(graph_list, seeds, label="Seeds", mc_runs=100):
    """Detailed empirical quantile and distribution analysis of a seed set."""
    spreads = [len(run_IC_simulation(graph_list, seeds)) for _ in range(mc_runs)]

    mean_val = np.mean(spreads)
    std_val = np.std(spreads)
    cv_val = (std_val / mean_val * 100.0) if mean_val > 0 else 0.0

    print(f"\n--- Cascade Heterogeneity: {label} (k={len(seeds)}, MC={mc_runs}) ---")
    print(f"Mean Spread (sigma)            : {mean_val:.2f}")
    print(f"Std Deviation (s)              : {std_val:.2f}")
    print(f"Coefficient of Variation (CV)  : {cv_val:.2f}%")
    print(f"Min .. Max Spread              : [{np.min(spreads)}, {np.max(spreads)}]")
    print(f"Quantiles [25%, 50%, 75%]      : [{np.percentile(spreads, 25):.1f}, {np.median(spreads):.1f}, {np.percentile(spreads, 75):.1f}]")

    print_ascii_histogram(spreads, label=label, bins=8)
    return mean_val, std_val, cv_val


def main():
    import ast
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hyperparams.yaml")
    ap.add_argument("--num-sets", type=int, default=30, help="Number of random seed sets to test ranking correlation")
    ap.add_argument("--mc-sims", type=int, default=20, help="MC simulations per set")
    ap.add_argument("--budget", type=int, default=20, help="Budget k to inspect cascade distribution")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_nodes, edge_index, graph_list = load_graph(cfg, device)
    dataset_name = cfg["dataset"].get("name", "Celegans")

    pmoe = PMoE(num_nodes, cfg["model"]["num_experts"], hidden_dim=cfg["model"]["hidden_dim"]).to(device)
    p_ckpt = f"checkpoints/pmoe_{dataset_name}.pth" if os.path.exists(f"checkpoints/pmoe_{dataset_name}.pth") else "checkpoints/pmoe.pth"

    if os.path.exists(p_ckpt):
        pmoe.load_state_dict(torch.load(p_ckpt, map_location=device))
        log.info(f"Loaded PMoE checkpoint from {p_ckpt}")
    else:
        log.warning(f"No checkpoint found at {p_ckpt}. Evaluating uninitialized model.")

    # 1. Ranking Correlation
    evaluate_ranking_correlation(pmoe, graph_list, edge_index, num_nodes, device, num_sets=args.num_sets, mc_sims=args.mc_sims)

    # 2. Side-by-Side Distribution: REM Seeds vs Degree Seeds
    from scripts.benchmark import get_degree_seeds
    deg_seeds = get_degree_seeds(graph_list, k=args.budget)

    csv_path = f"results/benchmark_{dataset_name}.csv"
    rem_seeds = None
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            row = df[df["Budget"] == args.budget]
            if not row.empty:
                rem_seeds = ast.literal_eval(str(row["Seeds"].values[0]))
        except Exception as e:
            log.warning(f"Could not load REM seeds from {csv_path}: {e}")

    print("\n" + "=" * 75)
    print(f"   CASCADE DISTRIBUTION COMPARISON [k = {args.budget} | Dataset: {dataset_name}]")
    print("=" * 75)

    if rem_seeds:
        analyze_cascade_heterogeneity(graph_list, rem_seeds, label="REM (Ours) Seeds", mc_runs=100)
    analyze_cascade_heterogeneity(graph_list, deg_seeds, label="Degree Heuristic Seeds", mc_runs=100)
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
