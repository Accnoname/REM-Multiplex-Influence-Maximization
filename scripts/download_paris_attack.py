"""
download_paris_attack.py — Prepare ParisAttack2015 dataset for REM.

The dataset must be downloaded manually due to server restrictions:

  1. Go to: https://manliodedomenico.com/data.php
  2. Find "ParisAttack2015" and download the ZIP
  3. Save it to: data/raw/paris_attack/ParisAttack2015_Multiplex_Social.zip
  4. Re-run this script to extract and verify.

Alternative:
  GitHub: https://github.com/manlius/SocialBursts
  Download the data from that repo's Data/ folder.

Usage:
    python scripts/download_paris_attack.py
"""
import os
import sys
import zipfile
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("REM")

RAW_DIR = "data/raw/paris_attack"
ZIP_NAME = "ParisAttack2015_Multiplex_Social.zip"
ZIP_PATH = os.path.join(RAW_DIR, ZIP_NAME)

# Expected filenames inside the zip (MuxViz format)
EXPECTED_EDGES = "paris_attack_multiplex.edges"
EXPECTED_NODES = "paris_attack_nodes.txt"


def extract_and_verify():
    if not os.path.exists(ZIP_PATH):
        log.error(f"ZIP not found at: {ZIP_PATH}")
        log.error("")
        log.error("Please download manually:")
        log.error("  1. Visit: https://manliodedomenico.com/data.php")
        log.error("  2. Download 'ParisAttack2015_Multiplex_Social'")
        log.error(f"  3. Save ZIP to: {ZIP_PATH}")
        log.error("  4. Re-run this script")
        sys.exit(1)

    os.makedirs(RAW_DIR, exist_ok=True)
    log.info(f"Found ZIP: {ZIP_PATH}")
    log.info("Extracting...")

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(RAW_DIR)
        names = z.namelist()

    log.info(f"Extracted {len(names)} files:")
    for n in names:
        fpath = os.path.join(RAW_DIR, n)
        size_mb = os.path.getsize(fpath) / 1e6 if os.path.isfile(fpath) else 0
        log.info(f"  {n}  ({size_mb:.1f} MB)")

    # Find edges and nodes files (auto-detect by extension)
    edges_file = next((n for n in names if ".edges" in n.lower()), None)
    nodes_file = next((n for n in names if "node" in n.lower() and ".txt" in n.lower()), None)

    log.info("")
    if edges_file:
        log.info(f"Detected edges file: {edges_file}")
        log.info(f"  -> Update configs/paris_attack.yaml: edges_file: \"{os.path.basename(edges_file)}\"")
    if nodes_file:
        log.info(f"Detected nodes file: {nodes_file}")
        log.info(f"  -> Update configs/paris_attack.yaml: nodes_file: \"{os.path.basename(nodes_file)}\"")

    log.info("")
    log.info("NEXT: Run preprocessing:")
    log.info("  python scripts/preprocess.py --config configs/paris_attack.yaml")


if __name__ == "__main__":
    extract_and_verify()
