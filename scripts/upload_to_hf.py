#!/usr/bin/env python3
"""
Upload Dual-Clock VLM Refusal Experiment Artifacts & Figures to Hugging Face Hub.

Uploads:
- outputs/figures/ (Figure 1, 2, 3 in PNG and PDF)
- outputs/final_verdict/ (final_verdict.json, summary)
- outputs/phase1/ (curves & results)
- outputs/validation_sweep/ (grid parquets)
- outputs/controls/ (controls JSON)
- logs/ (run logs)

Usage:
    # Uses logged-in HF credentials or HF_TOKEN env var:
    python scripts/upload_to_hf.py

    # Or specify custom repo and token:
    python scripts/upload_to_hf.py --repo-id <username>/dual-clock-vlm-refusal-results --token <hf_token>
"""

import argparse
import logging
import os
from pathlib import Path

from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Upload Experiment Artifacts to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, default=None, help="Target Hugging Face Dataset repo (e.g. username/repo-name)")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face API token")
    parser.add_argument("--private", action="store_true", help="Set repository to private")
    args = parser.parse_args()

    api = HfApi(token=args.token)

    try:
        user_info = api.whoami()
        user_name = user_info["name"]
        logger.info("Authenticated as Hugging Face user: %s", user_name)
    except Exception as exc:
        logger.error(
            "Authentication failed: %s. Please run 'huggingface-cli login' or pass --token <your_hf_token>.",
            exc,
        )
        return

    repo_id = args.repo_id or f"{user_name}/dual-clock-vlm-refusal-results"
    logger.info("Target Dataset Repository: https://huggingface.co/datasets/%s", repo_id)

    # 1. Ensure repository exists
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    logger.info("Dataset repository is ready.")

    # 2. Upload outputs folder (figures, final verdict, parquets)
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        logger.info("Uploading 'outputs' folder (figures, final verdict, parquets)...")
        api.upload_folder(
            folder_path=str(outputs_dir),
            path_in_repo="outputs",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload experimental outputs, figures, and final verdict",
        )
        logger.info("✓ Outputs uploaded successfully.")
    else:
        logger.warning("'outputs' directory not found.")

    # 3. Upload logs folder if present
    logs_dir = Path("logs")
    if logs_dir.exists():
        logger.info("Uploading 'logs' folder...")
        api.upload_folder(
            folder_path=str(logs_dir),
            path_in_repo="logs",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload experiment execution logs",
        )
        logger.info("✓ Logs uploaded successfully.")

    # 4. Upload docs/ if present
    docs_dir = Path("docs")
    if docs_dir.exists():
        logger.info("Uploading 'docs' folder...")
        api.upload_folder(
            folder_path=str(docs_dir),
            path_in_repo="docs",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload pre-registration and diagnostic documentation",
        )
        logger.info("✓ Documentation uploaded successfully.")

    print("\n" + "=" * 60)
    print(f"🎉 All artifacts successfully uploaded to:")
    print(f"👉 https://huggingface.co/datasets/{repo_id}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
