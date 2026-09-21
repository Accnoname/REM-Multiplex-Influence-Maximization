"""ablation.py — Systematic Component Ablation Study for REM.

Dissects and quantifies the empirical contribution of each architectural component:
1. Full REM (Seed2Vec + PMoE Latent Optimization + Local Search)
2. REM w/o Local Search (Pure Latent Gradient Ascent)
3. REM w/o Latent Optimization (Random Latent Vector z Decoded)
4. Degree Centrality Heuristic
5. Random Seed Baseline

Usage:
    python scripts/ablation.py --config configs/hyperparams.yaml --budget 20 --mc-runs 50
"""
import os
import sys
import yaml
import argparse
import random
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
from src.data.dataset import load_graph
from src.data.simulation import run_IC_simulation
from scripts.infer import robust_inference
from scripts.benchmark import get_degree_seeds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("Ablation")


def monte_carlo_eval(graph_list, seeds, mc_runs=50):
    spreads = [len(run_IC_simulation(graph_list, seeds)) for _ in range(mc_runs)]
    mean_spread = float(np.mean(spreads))
    std_spread = float(np.std(spreads))
    return mean_spread, std_spread


def run_ablation(config_path="configs/hyperparams.yaml", budget_k=20, mc_runs=50):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg["dataset"].get("name", "Celegans")
    _, _, graph_list = load_graph(cfg)
    all_nodes = list(graph_list[0].nodes())

    log.info(f"Running Ablation Study on [{dataset_name}] | Budget k={budget_k} | MC Runs={mc_runs}")

    variants = {}

    # Variant 1: Full REM
    log.info("1/5: Evaluating Full REM (Latent Gradient Ascent + Local Search)...")
    full_seeds = robust_inference(budget_k=budget_k, num_restarts=2, steps=50, config_path=config_path, enable_local_search=True, skip_optimization=False)
    m, s = monte_carlo_eval(graph_list, full_seeds, mc_runs=mc_runs)
    variants["Full REM"] = {"spread": m, "std": s, "seeds": full_seeds}

    # Variant 2: REM w/o Local Search
    log.info("2/6: Evaluating REM w/o Local Search (Pure Latent Ascent)...")
    no_ls_seeds = robust_inference(budget_k=budget_k, num_restarts=2, steps=50, config_path=config_path, enable_local_search=False, skip_optimization=False, disable_pmoe=False)
    m, s = monte_carlo_eval(graph_list, no_ls_seeds, mc_runs=mc_runs)
    variants["w/o Local Search"] = {"spread": m, "std": s, "seeds": no_ls_seeds}

    # Variant 3: REM w/o PMoE (Latent Ascent without PMoE guidance)
    log.info("3/6: Evaluating REM w/o PMoE Guidance (Latent Ascent purely on budget)...")
    no_pmoe_seeds = robust_inference(budget_k=budget_k, num_restarts=2, steps=50, config_path=config_path, enable_local_search=False, skip_optimization=False, disable_pmoe=True)
    m, s = monte_carlo_eval(graph_list, no_pmoe_seeds, mc_runs=mc_runs)
    variants["w/o PMoE Guidance"] = {"spread": m, "std": s, "seeds": no_pmoe_seeds}

    # Variant 4: REM w/o Optimization (Random z decoded via VAE)
    log.info("4/6: Evaluating REM w/o Latent Ascent (Random z decoded via VAE)...")
    no_opt_seeds = robust_inference(budget_k=budget_k, num_restarts=1, steps=0, config_path=config_path, enable_local_search=False, skip_optimization=True)
    m, s = monte_carlo_eval(graph_list, no_opt_seeds, mc_runs=mc_runs)
    variants["w/o Latent Ascent (Random z)"] = {"spread": m, "std": s, "seeds": no_opt_seeds}

    # Variant 5: Degree Centrality
    log.info("5/6: Evaluating Degree Centrality Heuristic...")
    deg_seeds = get_degree_seeds(graph_list, budget_k)
    m, s = monte_carlo_eval(graph_list, deg_seeds, mc_runs=mc_runs)
    variants["Degree Centrality"] = {"spread": m, "std": s, "seeds": deg_seeds}

    # Variant 6: Random Seeds
    log.info("6/6: Evaluating Random Baseline...")
    rand_seeds = random.sample(all_nodes, budget_k)
    m, s = monte_carlo_eval(graph_list, rand_seeds, mc_runs=mc_runs)
    variants["Random Seeds"] = {"spread": m, "std": s, "seeds": rand_seeds}

    # Build Summary Table
    rows = []
    base_spread = variants["Full REM"]["spread"]
    for name, data in variants.items():
        delta = ((data["spread"] - base_spread) / base_spread * 100.0) if base_spread > 0 else 0.0
        rows.append({
            "Component / Variant": name,
            "Mean Spread": f"{data['spread']:.2f}",
            "Std Dev": f"+/- {data['std']:.2f}",
            "Delta vs Full REM (%)": f"{delta:+.1f}%" if name != "Full REM" else "Reference (0.0%)"
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print(f"               REM ABLATION STUDY RESULTS [{dataset_name} | k={budget_k}]")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")

    out_csv = f"results/ablation_{dataset_name}_k{budget_k}.csv"
    os.makedirs("results", exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info(f"Saved ablation summary to {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hyperparams.yaml")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--mc-runs", type=int, default=30)
    args = ap.parse_args()

    run_ablation(config_path=args.config, budget_k=args.budget, mc_runs=args.mc_runs)
