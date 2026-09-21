"""preprocess.py — Build multiplex_graph.pkl + train_data.SG from raw edges file.

Usage:
    python scripts/preprocess.py                          # default: Celegans
    python scripts/preprocess.py --config configs/paris_attack.yaml
"""
import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from src.data.preprocessing import DataProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hyperparams.yaml")
    ap.add_argument("--skip-train-data", action="store_true",
                    help="Only build graph pkl, skip training data generation")
    args = ap.parse_args()

    dp = DataProcessor(config_path=args.config)
    dp.load_raw_graph()
    if not args.skip_train_data:
        dp.generate_training_data()


if __name__ == "__main__":
    main()
