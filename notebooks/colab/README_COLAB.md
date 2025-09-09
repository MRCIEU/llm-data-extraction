# Gemma3 270M Fine-tuning for Google Colab

This directory contains self-contained scripts for fine-tuning Gemma3 270M on Google Colab for MR abstract data extraction.

## Files

- `gemma3_270m_colab_finetune.py` - Self-contained training script for Colab
- `prepare-colab-dataset.py` - Data preparation script (in `scripts/python/preprocessing/`)
- `README_COLAB.md` - This documentation

## Quick Start for Google Colab

### Option 1: Use Sample Data (Fastest)

1. Upload `gemma3_270m_colab_finetune.py` to your Colab session
2. Run this in a Colab cell:

```python
# Load and run the script
exec(open('gemma3_270m_colab_finetune.py').read())

# Start fine-tuning with sample data
model, tokenizer = run_full_pipeline(use_sample_data=True)
```

### Option 2: Use Your Own Data

1. **Prepare your dataset locally** (run this on your local machine):
```bash
python scripts/python/preprocessing/prepare-colab-dataset.py \
    --data-dir data/intermediate \
    --output-file data/assets/colab/mr_dataset.json \
    --colab-package
```

2. **Upload files to Colab**:
   - Upload `gemma3_270m_colab_finetune.py`
   - Upload your dataset file (e.g., `mr_dataset.json`)

3. **Run in Colab**:
```python
# Load the script
exec(open('gemma3_270m_colab_finetune.py').read())

# Fine-tune with your data
model, tokenizer = run_full_pipeline(dataset_file="mr_dataset.json")
```

## Step-by-Step Colab Usage

### Cell 1: Install Dependencies
```python
# Load the script functions
exec(open('gemma3_270m_colab_finetune.py').read())

# Install required packages
install_dependencies()
```

### Cell 2: Import and Setup
```python
# Import all libraries
libs = import_libraries()

# Check GPU
check_gpu(libs)

# Load configuration
config = get_default_config()
print_config(config)
```

### Cell 3: Load Data
```python
# Option A: Use sample data
formatted_prompts = create_sample_dataset(libs)

# Option B: Load your dataset
# formatted_prompts = load_dataset_from_file("mr_dataset.json", libs)

print(f"Loaded {len(formatted_prompts)} training examples")

# Create HuggingFace dataset
train_dataset = libs['HFDataset'].from_dict({"text": formatted_prompts})
```

### Cell 4: Load Model
```python
# Load Gemma model with Unsloth
model, tokenizer = load_model_and_tokenizer(config, libs)
```

### Cell 5: Setup Training
```python
# Create trainer
trainer = create_trainer(model, tokenizer, train_dataset, config, libs)
```

### Cell 6: Train
```python
# Start training
train_model(trainer, libs)
```

### Cell 7: Test and Save
```python
# Test inference
test_inference(model, tokenizer, libs)

# Save model
save_model(model, tokenizer, config, libs)
```

## Configuration Options

You can customize the training by modifying the config:

```python
# Example: Faster training for testing
config = get_default_config()
config.update({
    'max_steps': 20,           # Fewer training steps
    'learning_rate': 1e-4,     # Lower learning rate
    'per_device_train_batch_size': 1,  # Smaller batch size
})

# Run with custom config
model, tokenizer = run_full_pipeline(
    dataset_file="your_data.json",
    config_overrides=config
)
```

## Key Parameters

### Model Configuration
- `model_name`: "unsloth/Gemma-2-2b" (placeholder for 270M)
- `max_seq_length`: 2048 (maximum sequence length)
- `load_in_4bit`: True (memory efficient quantization)

### Training Configuration
- `max_steps`: 60 (number of training steps)
- `learning_rate`: 2e-4 (learning rate)
- `per_device_train_batch_size`: 2 (batch size per GPU)
- `gradient_accumulation_steps`: 4 (effective batch size = 2 × 4 = 8)

### LoRA Configuration
- `r`: 16 (LoRA rank - controls parameter efficiency)
- `lora_alpha`: 16 (LoRA scaling factor)
- `target_modules`: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

## Data Format

Your dataset should be a JSON file with this structure:

```json
{
  "dataset_info": {
    "name": "mr_extraction_dataset",
    "num_examples": 100
  },
  "examples": [
    {
      "task_type": "metadata",
      "instruction": "Extract metadata from the following...",
      "response": "{\"title\": \"...\", \"authors\": [...]}",
      "abstract": "Background: ...",
      "source_pmid": "12345678"
    }
  ],
  "formatted_prompts": [
    "<bos><start_of_turn>user\nExtract metadata...<end_of_turn>\n<start_of_turn>model\n{...}<eos>"
  ]
}
```

## Memory Requirements

- **Minimum**: 8GB GPU memory (using 4-bit quantization)
- **Recommended**: 16GB GPU memory for stable training
- **Colab Free**: T4 GPU (16GB) should work with default settings
- **Colab Pro**: A100 or V100 for faster training

## Troubleshooting

### Memory Issues
```python
# Reduce memory usage
config.update({
    'per_device_train_batch_size': 1,
    'gradient_accumulation_steps': 8,  # Maintain effective batch size
    'max_seq_length': 1024,  # Shorter sequences
})
```

### Training Too Slow
```python
# Speed up training
config.update({
    'max_steps': 20,  # Fewer steps for testing
    'gradient_accumulation_steps': 2,  # Larger batch size if memory allows
})
```

### Installation Issues
```python
# Restart runtime and try again
import os
os.kill(os.getpid(), 9)  # Restart Colab runtime

# Or install manually
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" trl peft accelerate bitsandbytes
```

## Expected Outputs

After successful training, you'll have:

1. **Fine-tuned model**: Saved in `./gemma3-270m-mr-extraction/`
2. **GGUF model**: Quantized version for inference
3. **Training logs**: Showing loss reduction over time
4. **Test results**: Sample extractions to verify quality

## Performance Tips

1. **Start small**: Use `max_steps=20` for initial testing
2. **Monitor GPU**: Check `!nvidia-smi` to monitor memory usage
3. **Save checkpoints**: Model auto-saves during training
4. **Test early**: Run inference after 10-20 steps to check quality
5. **Adjust batch size**: Reduce if you get out-of-memory errors

## Integration with Your Workflow

This Colab fine-tuning integrates with your existing pipeline:

1. **Data source**: Uses your OpenAI extraction results as training data
2. **Same prompts**: Maintains compatibility with your prompt templates
3. **Same schemas**: Follows your existing JSON extraction schemas
4. **Local deployment**: Trained model can be used locally or in production

## Next Steps

After fine-tuning:

1. **Download the model**: Save to Google Drive or download locally
2. **Test on new data**: Validate extraction quality vs OpenAI
3. **Deploy for inference**: Use for batch processing of abstracts
4. **Iterate**: Fine-tune further with more data or different parameters

## Cost Comparison

- **OpenAI API**: ~$0.001-0.01 per abstract (depending on model)
- **Colab Pro**: ~$10/month for unlimited usage
- **Break-even**: ~1000-10000 abstracts per month

The fine-tuned model provides significant cost savings for large-scale extraction tasks while maintaining data privacy.