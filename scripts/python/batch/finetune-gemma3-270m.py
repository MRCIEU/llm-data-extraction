#!/usr/bin/env python3
"""
Finetune Gemma3 270M for MR abstract data extraction using Unsloth.

This script replicates the OpenAI-based extraction logic for MR abstracts
by fine-tuning a local Gemma3 270M model using LoRA/PEFT.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datasets import Dataset as HFDataset

# Add local packages to path
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent / "src" / "local_funcs" / "src")
)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "yiutils" / "src"))

from local_funcs import prompt_funcs, schema_funcs
from yiutils.failsafe import safe_json_loads


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning."""

    model_name: str = "unsloth/Gemma-2-2b"  # Will use 270M when available
    max_seq_length: int = 2048
    dtype: str = "bfloat16"
    load_in_4bit: bool = True

    # LoRA config
    r: int = 16
    target_modules: List[str] = None
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: dict = None

    # Training config
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 60
    learning_rate: float = 2e-4
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]


class MRExtractionDataset(Dataset):
    """Dataset for MR abstract extraction training."""

    def __init__(
        self,
        abstracts: List[str],
        metadata_extractions: List[str],
        results_extractions: List[str],
        tokenizer=None,
    ):
        self.abstracts = abstracts
        self.metadata_extractions = metadata_extractions
        self.results_extractions = results_extractions
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.abstracts)

    def __getitem__(self, idx):
        abstract = self.abstracts[idx]
        metadata_extraction = self.metadata_extractions[idx]
        results_extraction = self.results_extractions[idx]

        # Format as instruction-following examples
        metadata_instruction = self._format_instruction(abstract, "metadata")
        results_instruction = self._format_instruction(abstract, "results")

        return {
            "metadata_instruction": metadata_instruction,
            "metadata_response": metadata_extraction,
            "results_instruction": results_instruction,
            "results_response": results_extraction,
        }

    def _format_instruction(self, abstract: str, extraction_type: str) -> str:
        """Format instruction for the model."""
        if extraction_type == "metadata":
            instruction = (
                "Extract metadata from the following Mendelian randomization abstract. "
                "Return valid JSON with fields: title, authors, journal, year, pmid, doi, abstract.\n\n"
                f"Abstract: {abstract}\n\n"
                "Metadata JSON:"
            )
        else:  # results
            instruction = (
                "Extract MR results from the following Mendelian randomization abstract. "
                "Return valid JSON with fields: exposures, outcomes, mr_methods, effect_estimates, "
                "pvalues, confidence_intervals, sample_sizes, populations, study_design.\n\n"
                f"Abstract: {abstract}\n\n"
                "Results JSON:"
            )
        return instruction


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def load_training_data(data_dir: Path) -> Tuple[List[str], List[str], List[str]]:
    """Load training data from existing OpenAI extraction results."""
    logging.info(f"Loading training data from {data_dir}")

    abstracts = []
    metadata_extractions = []
    results_extractions = []

    # Look for processed results files
    for result_file in data_dir.glob("**/*results*.json"):
        try:
            with open(result_file, "r") as f:
                data = json.load(f)

            # Extract abstracts and extractions from OpenAI results
            if isinstance(data, list):
                for item in data:
                    if (
                        "abstract" in item
                        and "extracted_metadata" in item
                        and "extracted_results" in item
                    ):
                        abstracts.append(item["abstract"])
                        metadata_extractions.append(
                            json.dumps(item["extracted_metadata"])
                        )
                        results_extractions.append(
                            json.dumps(item["extracted_results"])
                        )

        except Exception as e:
            logging.warning(f"Failed to load {result_file}: {e}")
            continue

    logging.info(f"Loaded {len(abstracts)} training examples")
    return abstracts, metadata_extractions, results_extractions


def prepare_training_examples(
    abstracts: List[str],
    metadata_extractions: List[str],
    results_extractions: List[str],
) -> List[Dict[str, str]]:
    """Prepare training examples in instruction format."""
    examples = []

    for abstract, metadata, results in zip(
        abstracts, metadata_extractions, results_extractions
    ):
        # Metadata extraction example
        metadata_instruction = (
            "Extract metadata from the following Mendelian randomization abstract. "
            "Return valid JSON with fields: title, authors, journal, year, pmid, doi, abstract.\n\n"
            f"Abstract: {abstract}\n\n"
            "Metadata JSON:"
        )

        examples.append(
            {
                "instruction": metadata_instruction,
                "output": metadata,
            }
        )

        # Results extraction example
        results_instruction = (
            "Extract MR results from the following Mendelian randomization abstract. "
            "Return valid JSON with fields: exposures, outcomes, mr_methods, effect_estimates, "
            "pvalues, confidence_intervals, sample_sizes, populations, study_design.\n\n"
            f"Abstract: {abstract}\n\n"
            "Results JSON:"
        )

        examples.append(
            {
                "instruction": results_instruction,
                "output": results,
            }
        )

    return examples


def format_prompts(examples: List[Dict[str, str]]) -> List[str]:
    """Format examples as conversation prompts."""
    formatted = []

    for example in examples:
        # Use Gemma format
        prompt = f"<bos><start_of_turn>user\n{example['instruction']}<end_of_turn>\n<start_of_turn>model\n{example['output']}<eos>"
        formatted.append(prompt)

    return formatted


def load_unsloth_model(config: FineTuningConfig):
    """Load Gemma3 270M model with Unsloth."""
    try:
        from unsloth import FastLanguageModel
        import torch

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_seq_length,
            dtype=getattr(torch, config.dtype)
            if hasattr(torch, config.dtype)
            else None,
            load_in_4bit=config.load_in_4bit,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=config.r,
            target_modules=config.target_modules,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=config.bias,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            random_state=config.random_state,
            use_rslora=config.use_rslora,
            loftq_config=config.loftq_config,
        )

        return model, tokenizer

    except ImportError:
        logging.error("Unsloth not installed. Please install with: pip install unsloth")
        sys.exit(1)


def create_trainer(model, tokenizer, train_dataset, config: FineTuningConfig):
    """Create trainer for fine-tuning."""
    from transformers import TrainingArguments
    from trl import SFTTrainer

    training_args = TrainingArguments(
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        optim=config.optim,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        seed=config.seed,
        output_dir="./gemma3-270m-mr-extraction",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    return trainer


def evaluate_model(
    model, tokenizer, test_abstracts: List[str], config: FineTuningConfig
):
    """Evaluate the fine-tuned model on test examples."""
    logging.info("Evaluating model...")

    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)

    results = []

    for i, abstract in enumerate(test_abstracts[:5]):  # Test on first 5 examples
        # Test metadata extraction
        metadata_instruction = (
            "Extract metadata from the following Mendelian randomization abstract. "
            "Return valid JSON with fields: title, authors, journal, year, pmid, doi, abstract.\n\n"
            f"Abstract: {abstract}\n\n"
            "Metadata JSON:"
        )

        inputs = tokenizer(
            [
                f"<bos><start_of_turn>user\n{metadata_instruction}<end_of_turn>\n<start_of_turn>model\n"
            ],
            return_tensors="pt",
        ).to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.1,
            do_sample=True,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        metadata_extracted = response.split("<start_of_turn>model\n")[-1]

        # Test results extraction
        results_instruction = (
            "Extract MR results from the following Mendelian randomization abstract. "
            "Return valid JSON with fields: exposures, outcomes, mr_methods, effect_estimates, "
            "pvalues, confidence_intervals, sample_sizes, populations, study_design.\n\n"
            f"Abstract: {abstract}\n\n"
            "Results JSON:"
        )

        inputs = tokenizer(
            [
                f"<bos><start_of_turn>user\n{results_instruction}<end_of_turn>\n<start_of_turn>model\n"
            ],
            return_tensors="pt",
        ).to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.1,
            do_sample=True,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results_extracted = response.split("<start_of_turn>model\n")[-1]

        results.append(
            {
                "abstract": abstract[:200] + "...",
                "metadata_extraction": metadata_extracted,
                "results_extraction": results_extracted,
            }
        )

        logging.info(f"Evaluated example {i + 1}/5")

    return results


def save_model(model, tokenizer, output_dir: Path):
    """Save the fine-tuned model."""
    logging.info(f"Saving model to {output_dir}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Also save in GGUF format for inference
    try:
        from unsloth import FastLanguageModel

        model.save_pretrained_gguf(
            output_dir / "gguf", tokenizer, quantization_method="q4_k_m"
        )
        logging.info(f"Saved GGUF model to {output_dir / 'gguf'}")
    except Exception as e:
        logging.warning(f"Failed to save GGUF model: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    # ---- --data-dir ----
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/intermediate"),
        help="Directory containing training data",
    )

    # ---- --output-dir ----
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/gemma3-270m-mr-extraction"),
        help="Output directory for fine-tuned model",
    )

    # ---- --model-name ----
    parser.add_argument(
        "--model-name",
        type=str,
        default="unsloth/Gemma-2-2b",
        help="Base model name (will use 270M when available)",
    )

    # ---- --max-steps ----
    parser.add_argument(
        "--max-steps",
        type=int,
        default=60,
        help="Maximum training steps",
    )

    # ---- --learning-rate ----
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )

    # ---- --batch-size ----
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per device batch size",
    )

    # ---- --dry-run ----
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    setup_logging()

    if args.dry_run:
        logging.info("DRY RUN MODE - No actual training will be performed")

    # Create config
    config = FineTuningConfig(
        model_name=args.model_name,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
    )

    logging.info(f"Fine-tuning config: {config}")

    # Load training data
    abstracts, metadata_extractions, results_extractions = load_training_data(
        args.data_dir
    )

    if len(abstracts) == 0:
        logging.error("No training data found. Please run OpenAI extraction first.")
        sys.exit(1)

    # Prepare training examples
    examples = prepare_training_examples(
        abstracts, metadata_extractions, results_extractions
    )
    formatted_prompts = format_prompts(examples)

    # Create HuggingFace dataset
    train_dataset = HFDataset.from_dict({"text": formatted_prompts})

    logging.info(f"Prepared {len(formatted_prompts)} training examples")

    if args.dry_run:
        logging.info("DRY RUN: Would load model and start training here")
        logging.info(f"Training examples preview: {examples[0]}")
        return

    # Load model
    logging.info("Loading Unsloth model...")
    model, tokenizer = load_unsloth_model(config)

    # Create trainer
    trainer = create_trainer(model, tokenizer, train_dataset, config)

    # Train model
    logging.info("Starting training...")
    trainer.train()

    # Evaluate model
    test_abstracts = abstracts[-5:]  # Use last 5 as test
    evaluation_results = evaluate_model(model, tokenizer, test_abstracts, config)

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f, indent=2)

    # Save model
    save_model(model, tokenizer, args.output_dir)

    logging.info("Fine-tuning completed successfully!")
    logging.info(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
