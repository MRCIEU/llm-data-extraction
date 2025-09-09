#!/usr/bin/env python3
"""
Simple runner script for Gemma3 270M fine-tuning.

This script provides a command-line interface to run the fine-tuning
process with the configuration from the notebook.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def install_dependencies():
    """Install required packages for Unsloth."""
    print("Installing Unsloth and dependencies...")

    commands = [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "unsloth[colab-new]",
            "@",
            "git+https://github.com/unslothai/unsloth.git",
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "xformers<0.0.27",
            "trl",
            "peft",
            "accelerate",
            "bitsandbytes",
        ],
    ]

    for cmd in commands:
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies: {e}")
            return False

    return True


def check_gpu():
    """Check if GPU is available."""
    try:
        import torch

        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
            return True
        else:
            print("No GPU available. Fine-tuning will be slow on CPU.")
            return False
    except ImportError:
        print("PyTorch not installed. Please install it first.")
        return False


def run_training():
    """Run the training process by executing the notebook cells."""
    try:
        # Use the Python script version instead
        script_path = Path(__file__).parent / "finetune-gemma3-270m.py"
        if script_path.exists():
            print(f"Running training script: {script_path}")
            subprocess.run([sys.executable, str(script_path)], check=True)
        else:
            print("Training script not found. Please use the Jupyter notebook instead.")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        return False

    return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install required dependencies",
    )

    parser.add_argument(
        "--check-gpu",
        action="store_true",
        help="Check GPU availability",
    )

    parser.add_argument(
        "--run-training",
        action="store_true",
        help="Run the fine-tuning process",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Install deps, check GPU, and run training",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    if args.all:
        args.install_deps = True
        args.check_gpu = True
        args.run_training = True

    if args.install_deps:
        if not install_dependencies():
            sys.exit(1)

    if args.check_gpu:
        check_gpu()

    if args.run_training:
        if not run_training():
            sys.exit(1)

    if not any([args.install_deps, args.check_gpu, args.run_training, args.all]):
        print("No action specified. Use --help for options.")
        print("Quick start: python run_finetune.py --all")


if __name__ == "__main__":
    main()
