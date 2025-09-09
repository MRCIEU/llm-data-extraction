#!/usr/bin/env python3
"""
Self-contained Gemma3 270M Fine-tuning Script for Google Colab

This script contains all functions needed to fine-tune Gemma3 270M for MR abstract
data extraction. Designed to be run in Google Colab with minimal dependencies.

Usage:
1. Upload your dataset file to Colab
2. Run the installation cell
3. Configure parameters
4. Run training
"""

# ==== INSTALLATION AND IMPORTS ====


def install_dependencies():
    """Install required packages for Unsloth fine-tuning."""
    import subprocess
    import sys

    print("Installing Unsloth and dependencies...")

    # Install Unsloth
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        ]
    )

    # Install additional dependencies
    subprocess.check_call(
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
        ]
    )

    print("Installation completed!")


def import_libraries():
    """Import all required libraries."""
    import json
    import logging
    import os
    import sys
    from pathlib import Path
    from typing import Dict, List, Tuple, Any
    import numpy as np
    import torch
    import pandas as pd
    from datasets import Dataset as HFDataset
    from datetime import datetime

    # ML libraries
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments
    from trl import SFTTrainer

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    return {
        "json": json,
        "logging": logging,
        "os": os,
        "sys": sys,
        "Path": Path,
        "Dict": Dict,
        "List": List,
        "Tuple": Tuple,
        "Any": Any,
        "np": np,
        "torch": torch,
        "pd": pd,
        "HFDataset": HFDataset,
        "datetime": datetime,
        "FastLanguageModel": FastLanguageModel,
        "TrainingArguments": TrainingArguments,
        "SFTTrainer": SFTTrainer,
        "logger": logger,
    }


# ==== CONFIGURATION ====


def get_default_config():
    """Get default fine-tuning configuration."""
    return {
        "model_name": "unsloth/Gemma-2-2b",  # Will use 270M when available
        "max_seq_length": 2048,
        "dtype": "bfloat16",
        "load_in_4bit": True,
        # LoRA config
        "r": 16,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": 3407,
        # Training config
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 5,
        "max_steps": 60,
        "learning_rate": 2e-4,
        "fp16": False,
        "bf16": True,
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 3407,
        "output_dir": "./gemma3-270m-mr-extraction",
    }


# ==== DATA LOADING AND PROCESSING ====


def load_dataset_from_file(file_path, libs):
    """Load dataset from uploaded file."""
    json, logger = libs["json"], libs["logger"]

    logger.info(f"Loading dataset from {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.endswith(".json"):
                dataset = json.load(f)
                if "formatted_prompts" in dataset:
                    return dataset["formatted_prompts"]
                elif "examples" in dataset:
                    return format_examples_for_training(dataset["examples"], libs)
                else:
                    logger.error("Invalid dataset format")
                    return None
            elif file_path.endswith(".txt"):
                content = f.read()
                prompts = [p.strip() for p in content.split("\n\n") if p.strip()]
                return prompts
            else:
                logger.error("Unsupported file format")
                return None
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return None


def format_examples_for_training(examples, libs):
    """Format examples for Gemma training."""
    formatted_prompts = []

    for example in examples:
        if "instruction" in example and "response" in example:
            # Use Gemma conversation format
            formatted = (
                f"<bos><start_of_turn>user\n"
                f"{example['instruction']}"
                f"<end_of_turn>\n"
                f"<start_of_turn>model\n"
                f"{example['response']}"
                f"<eos>"
            )
            formatted_prompts.append(formatted)

    return formatted_prompts


def create_sample_dataset(libs):
    """Create a sample dataset for testing purposes."""
    json = libs["json"]

    sample_abstract = """
    Background: The relationship between body mass index (BMI) and cardiovascular disease 
    has been extensively studied, but the causal nature remains unclear. We used Mendelian 
    randomization to investigate the causal effect of BMI on coronary artery disease.
    
    Methods: We used genetic variants associated with BMI as instrumental variables in a 
    two-sample Mendelian randomization analysis. GWAS summary statistics were obtained 
    from the GIANT consortium (n=339,224) for BMI and CARDIoGRAMplusC4D (n=184,305) 
    for coronary artery disease.
    
    Results: Higher BMI was causally associated with increased risk of coronary artery 
    disease (OR = 1.27, 95% CI: 1.15-1.40, P = 2.3e-6). The effect remained significant 
    after adjustment for potential confounders.
    
    Conclusions: This study provides evidence for a causal relationship between higher 
    BMI and increased coronary artery disease risk.
    """

    # Sample metadata
    sample_metadata = {
        "title": "Causal relationship between BMI and coronary artery disease",
        "authors": ["Smith J", "Doe A"],
        "journal": "Nature Genetics",
        "year": 2023,
        "pmid": "12345678",
        "doi": "10.1038/ng.2023.001",
        "abstract": sample_abstract.strip(),
    }

    # Sample results
    sample_results = {
        "exposures": ["body mass index"],
        "outcomes": ["coronary artery disease"],
        "mr_methods": ["inverse variance weighted", "MR-Egger"],
        "effect_estimates": ["OR = 1.27"],
        "pvalues": ["2.3e-6"],
        "confidence_intervals": ["1.15-1.40"],
        "sample_sizes": ["339224", "184305"],
        "populations": ["European"],
        "study_design": "two-sample Mendelian randomization",
    }

    # Create training examples
    examples = []

    # Metadata example
    metadata_instruction = (
        "Extract metadata from the following Mendelian randomization abstract. "
        "Return valid JSON with fields: title, authors, journal, year, pmid, doi, abstract.\n\n"
        f"Abstract: {sample_abstract}\n\n"
        "Metadata JSON:"
    )

    examples.append(
        {
            "instruction": metadata_instruction,
            "response": json.dumps(sample_metadata, ensure_ascii=False),
        }
    )

    # Results example
    results_instruction = (
        "Extract MR results from the following Mendelian randomization abstract. "
        "Return valid JSON with fields: exposures, outcomes, mr_methods, effect_estimates, "
        "pvalues, confidence_intervals, sample_sizes, populations, study_design.\n\n"
        f"Abstract: {sample_abstract}\n\n"
        "Results JSON:"
    )

    examples.append(
        {
            "instruction": results_instruction,
            "response": json.dumps(sample_results, ensure_ascii=False),
        }
    )

    # Replicate examples for minimum training data
    extended_examples = examples * 20  # 40 examples total

    return format_examples_for_training(extended_examples, libs)


# ==== MODEL LOADING AND SETUP ====


def load_model_and_tokenizer(config, libs):
    """Load Gemma model with Unsloth for fine-tuning."""
    FastLanguageModel, torch, logger = (
        libs["FastLanguageModel"],
        libs["torch"],
        libs["logger"],
    )

    logger.info("Loading Unsloth model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        dtype=getattr(torch, config["dtype"])
        if hasattr(torch, config["dtype"])
        else None,
        load_in_4bit=config["load_in_4bit"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["r"],
        target_modules=config["target_modules"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias=config["bias"],
        use_gradient_checkpointing=config["use_gradient_checkpointing"],
        random_state=config["random_state"],
        use_rslora=False,
        loftq_config=None,
    )

    logger.info("Model loaded successfully!")
    return model, tokenizer


def create_trainer(model, tokenizer, train_dataset, config, libs):
    """Create trainer for fine-tuning."""
    TrainingArguments, SFTTrainer = libs["TrainingArguments"], libs["SFTTrainer"]

    training_args = TrainingArguments(
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        warmup_steps=config["warmup_steps"],
        max_steps=config["max_steps"],
        learning_rate=config["learning_rate"],
        fp16=config["fp16"],
        bf16=config["bf16"],
        logging_steps=config["logging_steps"],
        optim=config["optim"],
        weight_decay=config["weight_decay"],
        lr_scheduler_type=config["lr_scheduler_type"],
        seed=config["seed"],
        output_dir=config["output_dir"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    return trainer


# ==== TRAINING AND EVALUATION ====


def train_model(trainer, libs):
    """Train the model."""
    logger = libs["logger"]

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed!")


def evaluate_model(model, tokenizer, sample_abstracts, config, libs):
    """Evaluate the fine-tuned model."""
    FastLanguageModel, logger = libs["FastLanguageModel"], libs["logger"]

    logger.info("Evaluating model...")
    FastLanguageModel.for_inference(model)

    results = []

    for i, abstract in enumerate(sample_abstracts[:3]):
        print(f"\nEvaluating example {i + 1}/3...")

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

        print(f"Metadata extraction: {metadata_extracted[:200]}...")

        results.append(
            {
                "abstract": abstract[:200] + "...",
                "metadata_extraction": metadata_extracted,
            }
        )

    return results


def save_model(model, tokenizer, config, libs):
    """Save the fine-tuned model."""
    logger, Path = libs["logger"], libs["Path"]

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to {output_dir}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Try to save GGUF format
    try:
        model.save_pretrained_gguf(
            output_dir / "gguf", tokenizer, quantization_method="q4_k_m"
        )
        logger.info(f"Saved GGUF model to {output_dir / 'gguf'}")
    except Exception as e:
        logger.warning(f"Failed to save GGUF model: {e}")

    logger.info(f"Model saved successfully to {output_dir}")


# ==== INFERENCE TESTING ====


def test_inference(model, tokenizer, libs):
    """Test inference with a sample abstract."""
    FastLanguageModel = libs["FastLanguageModel"]

    sample_abstract = """
    Background: We investigated the causal effect of educational attainment on type 2 diabetes
    using Mendelian randomization. Methods: We used 74 genetic variants associated with educational
    attainment as instrumental variables. Results: Higher educational attainment was associated with
    lower risk of type 2 diabetes (OR = 0.82, 95% CI: 0.75-0.90, P < 0.001). Conclusions: Educational
    attainment has a protective causal effect on type 2 diabetes risk.
    """

    print("Testing inference on sample abstract...")
    print(f"Sample abstract: {sample_abstract[:150]}...")

    FastLanguageModel.for_inference(model)

    # Test metadata extraction
    instruction = (
        "Extract metadata from the following Mendelian randomization abstract. "
        "Return valid JSON with fields: title, authors, journal, year, pmid, doi, abstract.\n\n"
        f"Abstract: {sample_abstract}\n\n"
        "Metadata JSON:"
    )

    inputs = tokenizer(
        [
            f"<bos><start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
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
    result = response.split("<start_of_turn>model\n")[-1]

    print(f"\nExtraction result:\n{result}")
    return result


# ==== UTILITY FUNCTIONS ====


def check_gpu(libs):
    """Check GPU availability and memory."""
    torch = libs["torch"]

    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
        return True
    else:
        print("❌ No GPU available. Training will be very slow.")
        return False


def print_config(config):
    """Print configuration in a readable format."""
    print("🔧 Training Configuration:")
    print(f"  Model: {config['model_name']}")
    print(f"  Max sequence length: {config['max_seq_length']}")
    print(f"  LoRA rank: {config['r']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Batch size: {config['per_device_train_batch_size']}")
    print(f"  Max steps: {config['max_steps']}")
    print(f"  Output directory: {config['output_dir']}")


# ==== MAIN EXECUTION FUNCTIONS ====


def run_full_pipeline(dataset_file=None, use_sample_data=False, config_overrides=None):
    """Run the complete fine-tuning pipeline."""

    # Step 1: Install dependencies
    print("🚀 Installing dependencies...")
    install_dependencies()

    # Step 2: Import libraries
    print("📚 Importing libraries...")
    libs = import_libraries()
    logger = libs["logger"]
    HFDataset = libs["HFDataset"]

    # Step 3: Check GPU
    print("🔍 Checking GPU...")
    check_gpu(libs)

    # Step 4: Load configuration
    print("⚙️ Loading configuration...")
    config = get_default_config()
    if config_overrides:
        config.update(config_overrides)
    print_config(config)

    # Step 5: Load dataset
    print("📊 Loading dataset...")
    if use_sample_data or not dataset_file:
        logger.info("Using sample dataset for demonstration")
        formatted_prompts = create_sample_dataset(libs)
    else:
        formatted_prompts = load_dataset_from_file(dataset_file, libs)
        if not formatted_prompts:
            logger.error("Failed to load dataset, falling back to sample data")
            formatted_prompts = create_sample_dataset(libs)

    print(f"Loaded {len(formatted_prompts)} training examples")

    # Create HuggingFace dataset
    train_dataset = HFDataset.from_dict({"text": formatted_prompts})

    # Step 6: Load model
    print("🤖 Loading model...")
    model, tokenizer = load_model_and_tokenizer(config, libs)

    # Step 7: Create trainer
    print("🏋️ Setting up trainer...")
    trainer = create_trainer(model, tokenizer, train_dataset, config, libs)

    # Step 8: Train model
    print("🎯 Starting training...")
    train_model(trainer, libs)

    # Step 9: Test inference
    print("🧪 Testing inference...")
    test_inference(model, tokenizer, libs)

    # Step 10: Save model
    print("💾 Saving model...")
    save_model(model, tokenizer, config, libs)

    print("✅ Fine-tuning completed successfully!")
    print(f"Model saved to: {config['output_dir']}")

    return model, tokenizer


def run_with_custom_config():
    """Example of running with custom configuration."""
    custom_config = {
        "max_steps": 30,  # Shorter training for testing
        "learning_rate": 1e-4,  # Lower learning rate
        "per_device_train_batch_size": 1,  # Smaller batch size for limited GPU
    }

    return run_full_pipeline(use_sample_data=True, config_overrides=custom_config)


# ==== EXAMPLE USAGE ====

if __name__ == "__main__":
    # Example 1: Run with sample data
    print("Running fine-tuning with sample data...")
    model, tokenizer = run_full_pipeline(use_sample_data=True)

    # Example 2: Run with uploaded dataset (uncomment and modify path)
    # print("Running fine-tuning with uploaded dataset...")
    # model, tokenizer = run_full_pipeline(dataset_file="mr_extraction_dataset.json")

    # Example 3: Run with custom configuration
    # print("Running fine-tuning with custom config...")
    # model, tokenizer = run_with_custom_config()


# ==== COLAB ADAPTATION NOTES ====
"""
To adapt this script for Google Colab:

1. Upload this script to Colab
2. Upload your dataset file (if you have one)
3. Run the cells in order:

# Cell 1: Run the full pipeline
exec(open('gemma3_270m_colab_finetune.py').read())

# Cell 2: Quick start with sample data
model, tokenizer = run_full_pipeline(use_sample_data=True)

# Cell 3: With your own dataset
# model, tokenizer = run_full_pipeline(dataset_file="your_dataset.json")

# Cell 4: Test the trained model
test_inference(model, tokenizer, import_libraries())

Alternative: Copy-paste individual functions into Colab cells and run step by step.
"""
