"""benchmark.py — Comprehensive REM benchmarking with advanced evaluation metrics and baseline comparisons."""
import sys
import os
import ast
import yaml
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
from src.data.dataset import load_graph
from src.data.simulation import run_IC_simulation
from scripts.infer import robust_inference

log = logging.getLogger("REM")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def monte_carlo(graph_list, seeds, n=100):
    """Run n IC simulations, return (mean, std, se, cv)."""
    spreads = [len(run_IC_simulation(graph_list, seeds)) for _ in tqdm(range(n), leave=False)]
    mean = float(np.mean(spreads))
    std = float(np.std(spreads))
    se = float(std / np.sqrt(n))
    cv = float((std / mean * 100.0) if mean > 0 else 0.0)
    return mean, std, se, cv


def get_degree_seeds(graph_list, k):
    """Multi-layer aggregated degree centrality baseline."""
    deg = {}
    for G in graph_list:
        for n in G.nodes():
            deg[n] = deg.get(n, 0) + G.degree(n)
    return sorted(deg.keys(), key=lambda n: deg[n], reverse=True)[:k]


def run_random_baseline(graph_list, k, n_seeds=5, n_mc=20):
    """Average spread over multiple random seed sets."""
    all_nodes = list(graph_list[0].nodes())
    all_spreads = []
    for _ in range(n_seeds):
        r_seeds = random.sample(all_nodes, k)
        for _ in range(n_mc):
            all_spreads.append(len(run_IC_simulation(graph_list, r_seeds)))
    mean = float(np.mean(all_spreads))
    std = float(np.std(all_spreads))
    se = float(std / np.sqrt(len(all_spreads)))
    cv = float((std / mean * 100.0) if mean > 0 else 0.0)
    return mean, std, se, cv


def run_benchmark(config_path="configs/hyperparams.yaml", recompute_rem=False, mc_runs=100):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _, _, graph_list = load_graph(cfg)
    dataset_name = cfg["dataset"].get("name", "dataset")
    budgets = [10, 20, 30, 40, 50]
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    # Paper reported baseline for Celegans (AAAI 2025 / comparative benchmark)
    paper_baseline = {10: 105.3, 20: 210.5, 30: 295.2, 40: 370.8, 50: 426.7} if dataset_name == "Celegans" else {}

    # Check for existing REM seeds
    existing_csv = os.path.join(out_dir, f"benchmark_{dataset_name}.csv")
    cached_seeds = {}
    if os.path.exists(existing_csv) and not recompute_rem:
        try:
            prev_df = pd.read_csv(existing_csv)
            if "Budget" in prev_df.columns and "Seeds" in prev_df.columns:
                for _, r in prev_df.iterrows():
                    cached_seeds[int(r["Budget"])] = ast.literal_eval(str(r["Seeds"]))
                log.info(f"Loaded cached REM seeds from {existing_csv}")
        except Exception as e:
            log.warning(f"Could not read existing cache: {e}")

    log.info(f"Running Advanced Benchmark [{dataset_name}] | budgets={budgets} | MC runs={mc_runs}")

    rows = []
    comparison_rows = []

    for k in budgets:
        log.info(f"--- Evaluating Budget k = {k} ---")

        # 1. REM (Ours)
        if k in cached_seeds and not recompute_rem:
            rem_seeds = cached_seeds[k]
            log.info(f"[REM] Using cached seeds ({len(rem_seeds)})")
        else:
            log.info(f"[REM] Running robust inference...")
            rem_seeds = robust_inference(budget_k=k, num_restarts=3, steps=60, config_path=config_path)

        rem_mean, rem_std, rem_se, rem_cv = monte_carlo(graph_list, rem_seeds, n=mc_runs)
        rem_eff = rem_mean / k

        # 2. Degree Baseline
        deg_seeds = get_degree_seeds(graph_list, k)
        deg_mean, deg_std, deg_se, deg_cv = monte_carlo(graph_list, deg_seeds, n=mc_runs)
        deg_eff = deg_mean / k

        # Jaccard overlap between REM and Degree
        rem_set, deg_set = set(rem_seeds), set(deg_seeds)
        overlap = len(rem_set.intersection(deg_set))
        jaccard = overlap / len(rem_set.union(deg_set)) if len(rem_set.union(deg_set)) > 0 else 0.0

        # 3. Random Baseline
        rand_mean, rand_std, rand_se, rand_cv = run_random_baseline(graph_list, k, n_seeds=5, n_mc=20)
        rand_eff = rand_mean / k

        # 4. Paper Benchmark
        paper_ref = paper_baseline.get(k, np.nan)
        nsr = (rem_mean / paper_ref * 100.0) if paper_ref > 0 else np.nan
        gain_over_rand = ((rem_mean - rand_mean) / rand_mean * 100.0) if rand_mean > 0 else 0.0
        gain_over_paper = ((rem_mean - paper_ref) / paper_ref * 100.0) if paper_ref > 0 else np.nan

        log.info(f"k={k} | REM: {rem_mean:.2f}+/-{rem_std:.2f} | Paper: {paper_ref:.1f} | Deg: {deg_mean:.2f} | Rand: {rand_mean:.2f}")

        # Store detailed row
        rows.append({
            "Budget": k,
            "REM_Spread": round(rem_mean, 2),
            "REM_Std": round(rem_std, 2),
            "REM_SE": round(rem_se, 2),
            "REM_CV_Pct": round(rem_cv, 2),
            "REM_Efficiency": round(rem_eff, 2),
            "Degree_Spread": round(deg_mean, 2),
            "Degree_Std": round(deg_std, 2),
            "Degree_Efficiency": round(deg_eff, 2),
            "Random_Spread": round(rand_mean, 2),
            "Random_Std": round(rand_std, 2),
            "Paper_Baseline": round(paper_ref, 2),
            "NSR_vs_Paper_Pct": round(nsr, 2),
            "Gain_vs_Random_Pct": round(gain_over_rand, 2),
            "Gain_vs_Paper_Pct": round(gain_over_paper, 2),
            "Seed_Overlap": overlap,
            "Jaccard_Deg": round(jaccard, 4),
            "Seeds": str(rem_seeds)
        })

        comparison_rows.append({
            "Budget (k)": k,
            "Random": f"{rand_mean:.1f} +/- {rand_std:.1f}",
            "Paper Ref": f"{paper_ref:.1f}",
            "REM (Ours)": f"{rem_mean:.1f} +/- {rem_std:.1f}",
            "Gain vs Random": f"+{gain_over_rand:.1f}%",
            "NSR vs Paper": f"{nsr:.1f}%",
            "REM Eff (spread/k)": f"{rem_eff:.2f}",
            "Jaccard (REM vs Deg)": f"{jaccard:.3f}"
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"benchmark_{dataset_name}.csv")
    df.to_csv(csv_path, index=False)
    log.info(f"Saved full benchmark to {csv_path}")

    comp_df = pd.DataFrame(comparison_rows)
    comp_csv_path = os.path.join(out_dir, f"metrics_comparison_{dataset_name}.csv")
    comp_df.to_csv(comp_csv_path, index=False)
    log.info(f"Saved comparison summary to {comp_csv_path}")

    # Plot 2-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Spread Comparison
    ax1.plot(df["Budget"], df["REM_Spread"], marker="o", linewidth=2.5, color="#1f77b4", label="REM (Ours)")
    ax1.fill_between(
        df["Budget"],
        df["REM_Spread"] - df["REM_Std"],
        df["REM_Spread"] + df["REM_Std"],
        color="#1f77b4", alpha=0.15, label="REM +/- 1 Std Dev"
    )
    ax1.plot(df["Budget"], df["Paper_Baseline"], marker="s", linestyle="--", color="#d62728", linewidth=2, label="Paper Benchmark")
    ax1.plot(df["Budget"], df["Random_Spread"], marker="^", linestyle=":", color="#7f7f7f", linewidth=1.8, label="Random Seeds")
    ax1.plot(df["Budget"], df["Degree_Spread"], marker="d", linestyle="-.", color="#2ca02c", linewidth=1.5, label="Degree Heuristic")

    ax1.set_title(f"Influence Spread Comparison ({dataset_name})", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Seed Budget (k)", fontsize=11)
    ax1.set_ylabel("Expected Spread sigma(S)", fontsize=11)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Spread Efficiency & Relative Gain
    ax2.plot(df["Budget"], df["REM_Efficiency"], marker="o", color="#1f77b4", linewidth=2, label="REM Efficiency (spread/k)")
    if not df["Paper_Baseline"].isna().all():
        ax2.plot(df["Budget"], df["Paper_Baseline"] / df["Budget"], marker="s", linestyle="--", color="#d62728", linewidth=1.8, label="Paper Efficiency (spread/k)")
    ax2.plot(df["Budget"], df["Random_Spread"] / df["Budget"], marker="^", linestyle=":", color="#7f7f7f", linewidth=1.5, label="Random Efficiency")

    ax2_twin = ax2.twinx()
    if not df["NSR_vs_Paper_Pct"].isna().all():
        ax2_twin.plot(df["Budget"], df["NSR_vs_Paper_Pct"], marker="x", color="#ff7f0e", linestyle="-", linewidth=2, label="NSR vs Paper (%)")
        ax2_twin.axhline(100.0, color="#ff7f0e", linestyle=":", alpha=0.7)
        ax2_twin.set_ylabel("Normalized Spread Ratio NSR (%)", color="#ff7f0e", fontsize=11)
    else:
        ax2_twin.plot(df["Budget"], df["Gain_vs_Random_Pct"], marker="x", color="#ff7f0e", linestyle="-", linewidth=2, label="Gain vs Random (%)")
        ax2_twin.set_ylabel("Gain vs Random (%)", color="#ff7f0e", fontsize=11)
    ax2_twin.tick_params(axis="y", labelcolor="#ff7f0e")

    ax2.set_title("Spread Efficiency & Relative Gain", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Seed Budget (k)", fontsize=11)
    ax2.set_ylabel("Efficiency (Spread per Seed Node)", fontsize=11)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_path = os.path.join(out_dir, f"chart_{dataset_name}.png")
    fig.savefig(chart_path, dpi=300)
    plt.close()
    log.info(f"Saved comparison chart to {chart_path}")

    # Display console summary
    print("\n" + "=" * 85)
    print(f"       REM BENCHMARK & METRICS COMPARISON SUMMARY [{dataset_name}]")
    print("=" * 85)
    print(comp_df.to_string(index=False))
    print("=" * 85 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hyperparams.yaml", help="Path to config yaml")
    ap.add_argument("--recompute-rem", action="store_true", help="Re-run latent gradient ascent for REM seeds")
    ap.add_argument("--mc-runs", type=int, default=100, help="Number of Monte Carlo simulations")
    args = ap.parse_args()

    run_benchmark(config_path=args.config, recompute_rem=args.recompute_rem, mc_runs=args.mc_runs)
