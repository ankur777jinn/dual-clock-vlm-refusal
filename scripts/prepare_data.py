#!/usr/bin/env python3
"""
Data preparation utility for dual-clock-vlm-refusal.

Features:
1. Downloads standard AdvBench (520 harmful prompts) and Alpaca (520 benign prompts).
2. Generates or validates the `manifest.csv` required by `src.data_loader.DatasetManager`.
3. Optionally generates synthetic/dummy images for a rapid smoke test.

Usage:
    # 1. Download text datasets:
    python scripts/prepare_data.py --download-text --data-dir data

    # 2. Generate dummy dataset for end-to-end smoke testing:
    python scripts/prepare_data.py --create-dummy --data-dir data
"""

import argparse
import csv
import logging
import urllib.request
from pathlib import Path
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"


def download_advbench(output_path: Path):
    """Download and extract harmful goal strings from AdvBench CSV."""
    logger.info("Downloading AdvBench harmful prompts...")
    import io
    req = urllib.request.Request(ADVBENCH_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        content = resp.read().decode("utf-8")
    
    reader = csv.DictReader(io.StringIO(content))
    goals = [row["goal"].strip() for row in reader if row.get("goal")]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for g in goals[:520]:
            f.write(f"{g}\n")
    logger.info("Saved %d AdvBench prompts to %s", len(goals[:520]), output_path)


def download_alpaca(output_path: Path):
    """Download and extract benign instruction strings from Stanford Alpaca JSON."""
    logger.info("Downloading Alpaca benign prompts...")
    import json
    req = urllib.request.Request(ALPACA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    
    instructions = [item["instruction"].strip() for item in data if item.get("instruction")]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for inst in instructions[:520]:
            f.write(f"{inst}\n")
    logger.info("Saved %d Alpaca prompts to %s", len(instructions[:520]), output_path)


def create_dummy_dataset(data_dir: Path):
    """Create placeholder images and manifest.csv for pipeline testing."""
    logger.info("Creating dummy images and manifest.csv in %s...", data_dir)
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_rows = []
    
    # 1. Natural images: 200 HS, 200 SS (total 400)
    # Splits: Tuning (50 HS, 50 SS, 20 pairs)
    #         Validation (50 HS, 50 SS)
    #         Test (100 HS, 100 SS, 40 pairs)
    splits_spec = [
        ("tuning", 50, 20),
        ("validation", 50, 0),
        ("test", 100, 40),
    ]
    
    sample_counter = 0
    for split_name, n_per_cohort, n_pairs in splits_spec:
        for i in range(n_per_cohort):
            pair_id = f"pair_{split_name}_{i}" if i < n_pairs else ""
            
            # HS image (red tinted dummy)
            hs_id = f"hs_{sample_counter:04d}"
            hs_rel_path = f"images/{hs_id}.png"
            hs_img = Image.new("RGB", (448, 448), color=(200, 50, 50))
            d = ImageDraw.Draw(hs_img)
            d.text((50, 200), f"HS Hazard: {hs_id}", fill=(255, 255, 255))
            hs_img.save(data_dir / hs_rel_path)
            manifest_rows.append({"id": hs_id, "path": hs_rel_path, "cohort": "hs", "split": split_name, "pair_id": pair_id})
            
            # SS image (green tinted dummy)
            ss_id = f"ss_{sample_counter:04d}"
            ss_rel_path = f"images/{ss_id}.png"
            ss_img = Image.new("RGB", (448, 448), color=(50, 150, 50))
            d = ImageDraw.Draw(ss_img)
            d.text((50, 200), f"SS Safe: {ss_id}", fill=(255, 255, 255))
            ss_img.save(data_dir / ss_rel_path)
            manifest_rows.append({"id": ss_id, "path": ss_rel_path, "cohort": "ss", "split": split_name, "pair_id": pair_id})
            
            sample_counter += 1
            
    # 2. Control images: 50 plumbing, 50 gardening (Test split)
    for i in range(50):
        c_id = f"ctrl_plumb_{i:03d}"
        c_rel_path = f"images/{c_id}.png"
        img = Image.new("RGB", (448, 448), color=(50, 50, 200))
        img.save(data_dir / c_rel_path)
        manifest_rows.append({"id": c_id, "path": c_rel_path, "cohort": "control_plumbing", "split": "test", "pair_id": ""})
        
    for i in range(50):
        c_id = f"ctrl_garden_{i:03d}"
        c_rel_path = f"images/{c_id}.png"
        img = Image.new("RGB", (448, 448), color=(100, 150, 100))
        img.save(data_dir / c_rel_path)
        manifest_rows.append({"id": c_id, "path": c_rel_path, "cohort": "control_gardening", "split": "test", "pair_id": ""})
        
    manifest_path = data_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "path", "cohort", "split", "pair_id"])
        writer.writeheader()
        writer.writerows(manifest_rows)
        
    logger.info("Created %d dummy images and wrote %s", len(manifest_rows), manifest_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset and prompts for dual-clock refusal")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--download-text", action="store_true", help="Download AdvBench and Alpaca text prompts")
    parser.add_argument("--create-dummy", action="store_true", help="Create dummy image dataset for smoke testing")
    args = parser.parse_args()
    
    args.data_dir.mkdir(parents=True, exist_ok=True)
    
    if args.download_text or not (args.data_dir / "advbench_520.txt").exists():
        download_advbench(args.data_dir / "advbench_520.txt")
        download_alpaca(args.data_dir / "alpaca_520.txt")
        
    if args.create_dummy:
        create_dummy_dataset(args.data_dir)


if __name__ == "__main__":
    main()
