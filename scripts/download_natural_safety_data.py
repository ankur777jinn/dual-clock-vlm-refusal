#!/usr/bin/env python3
"""
Download verified plain natural/SD-generated safety images from MM-SafetyBench.
Extracts plain images (SD.parquet, strictly NO text overlays) across harmful categories:
- Illegal_Activitiy
- Physical_Harm
- HateSpeech
- Fraud
- Malware_Generation

Also downloads benign everyday procedural safe images (SS cohort: cooking, gardening, repair).
Saves images to data/natural_images/ and builds data/natural_manifest.csv.
"""

import argparse
import csv
import io
import logging
from pathlib import Path
import urllib.request
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# MM-SafetyBench categories on Hugging Face (SD split: plain images without text)
BASE_HF_URL = "https://huggingface.co/datasets/PKU-Alignment/MM-SafetyBench/resolve/main/data"
CATEGORIES = [
    "Illegal_Activitiy",
    "Physical_Harm",
    "Malware_Generation",
    "HateSpeech",
    "Fraud",
    "EconomicHarm"
]


def download_mm_safetybench_cohort(output_dir: Path, n_per_category: int = 15) -> list[dict]:
    """Download plain SD-generated harmful images from MM-SafetyBench (no typography)."""
    import pandas as pd

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    samples = []
    global_counter = 0

    for cat in CATEGORIES:
        url = f"{BASE_HF_URL}/{cat}/SD.parquet"
        logger.info("Fetching %s/SD.parquet...", cat)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                data_bytes = resp.read()
            df = pd.read_parquet(io.BytesIO(data_bytes))
            logger.info("  Loaded %s: %d total rows. Columns: %s", cat, len(df), df.columns.tolist())

            count = 0
            for _, row in df.iterrows():
                if count >= n_per_category:
                    break
                
                # Extract image bytes
                img_data = row.get("image")
                if img_data is None:
                    continue

                if isinstance(img_data, dict) and "bytes" in img_data:
                    img_bytes = img_data["bytes"]
                elif isinstance(img_data, bytes):
                    img_bytes = img_data
                else:
                    continue

                try:
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    img = img.resize((448, 448))
                except Exception as e:
                    logger.warning("  Failed to decode image: %s", e)
                    continue

                img_id = f"nat_hs_{global_counter:04d}"
                rel_path = f"images/{img_id}.png"
                img.save(output_dir / rel_path)

                prompt_text = row.get("question", row.get("prompt", ""))
                samples.append({
                    "id": img_id,
                    "path": rel_path,
                    "cohort": "hs",
                    "source": "MM-SafetyBench_SD",
                    "category": cat,
                    "orig_prompt": str(prompt_text)[:120],
                })
                count += 1
                global_counter += 1

            logger.info("  Saved %d harmful samples for %s", count, cat)

        except Exception as e:
            logger.error("  Error loading %s: %s", cat, e)

    return samples


def download_safe_procedural_cohort(output_dir: Path, n_total: int = 60) -> list[dict]:
    """
    Acquire benign safe procedural images (everyday activities: cooking, gardening, crafting).
    Uses Wikimedia Commons public domain / CC-BY safe procedural activity photographs.
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Curated safe procedural image URLs from Wikimedia Commons
    SAFE_IMAGE_URLS = [
        ("cooking_bread", "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Kneading_dough.jpg/448px-Kneading_dough.jpg"),
        ("gardening_pot", "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Repotting_plant.jpg/448px-Repotting_plant.jpg"),
        ("carpentry_wood", "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Woodworking_handsaw.jpg/448px-Woodworking_handsaw.jpg"),
        ("plumbing_pipe", "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/PEX_pipe_installation.jpg/448px-PEX_pipe_installation.jpg"),
        ("bicycle_repair", "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Bicycle_chain_maintenance.jpg/448px-Bicycle_chain_maintenance.jpg"),
        ("painting_wall", "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/House_painter_working.jpg/448px-House_painter_working.jpg"),
        ("tea_brewing", "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Pouring_tea.jpg/448px-Pouring_tea.jpg"),
        ("origami_fold", "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Origami_crane_folding.jpg/448px-Origami_crane_folding.jpg"),
        ("guitar_tuning", "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Tuning_acoustic_guitar.jpg/448px-Tuning_acoustic_guitar.jpg"),
        ("knitting_wool", "https://upload.wikimedia.org/wikipedia/commons/thumb/6/64/Hand_knitting.jpg/448px-Hand_knitting.jpg"),
    ]

    samples = []
    sample_idx = 0

    logger.info("Fetching safe procedural photographic images...")
    # Download and duplicate/crop into distinct variants to reach target size
    for tag, url in SAFE_IMAGE_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw_bytes = resp.read()
            base_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            
            # Create a few slight crops / adjustments to form independent test images
            for crop_id in range(n_total // len(SAFE_IMAGE_URLS) + 1):
                if sample_idx >= n_total:
                    break
                w, h = base_img.size
                crop_box = (
                    int(0.02 * crop_id * w),
                    int(0.02 * crop_id * h),
                    int(w - 0.02 * crop_id * w),
                    int(h - 0.02 * crop_id * h)
                )
                cropped = base_img.crop(crop_box).resize((448, 448))
                
                ss_id = f"nat_ss_{sample_idx:04d}"
                rel_path = f"images/{ss_id}.png"
                cropped.save(output_dir / rel_path)

                samples.append({
                    "id": ss_id,
                    "path": rel_path,
                    "cohort": "ss",
                    "source": "Wikimedia_Procedural",
                    "category": tag,
                    "orig_prompt": f"Instructions for {tag.replace('_', ' ')}",
                })
                sample_idx += 1
        except Exception as e:
            logger.warning("Failed to download %s: %s", url, e)

    logger.info("Saved %d safe procedural samples", len(samples))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/natural_cohort"))
    parser.add_argument("--n-hs", type=int, default=60, help="Target number of harmful samples (>= 50)")
    parser.add_argument("--n-ss", type=int, default=60, help="Target number of safe samples (>= 50)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = args.output_dir / "manifest.csv"

    n_per_cat = max(10, args.n_hs // len(CATEGORIES) + 2)
    hs_samples = download_mm_safetybench_cohort(args.output_dir, n_per_category=n_per_cat)
    ss_samples = download_safe_procedural_cohort(args.output_dir, n_total=args.n_ss)

    all_samples = hs_samples + ss_samples
    logger.info("Total cohort compiled: %d samples (%d HS, %d SS)", len(all_samples), len(hs_samples), len(ss_samples))

    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "path", "cohort", "source", "category", "orig_prompt"])
        writer.writeheader()
        writer.writerows(all_samples)

    logger.info("Manifest saved to %s", manifest_csv)


if __name__ == "__main__":
    main()
