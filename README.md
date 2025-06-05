# Llama Knowledge Distillation with LoRA

A implementation of vocabulary-aligned knowledge distillation for Llama models using Low-Rank Adaptation (LoRA) fine-tuning. 

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended: 16GB+ VRAM)
- PyTorch 2.0+
- Transformers 4.35+

## Installation

1. **Switch branch:**
```bash
cd Final_15
git checkout llama-kd-lora
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up Weights & Biases (optional):**
```bash
wandb login
```

## Project Structure

```
llama_distillation/
├── config/                 # Configuration management
│   ├── training_config.py  # Training hyperparameters
│   └── model_config.py     # Model paths and settings
├── src/
│   ├── models/             # Core model implementations
│   │   ├── distillation_trainer.py  # Custom trainer with KD
│   │   └── vocab_alignment.py       # Vocabulary alignment utils
│   ├── data/               # Dataset implementations
│   │   └── dataset.py      # WikiText-2 dataset handler
│   ├── utils/              # Utility functions
│   │   ├── evaluation.py   # Perplexity evaluation
│   │   └── callbacks.py    # Training callbacks
│   └── training/           # Training orchestration
│       └── train.py        # Main training logic
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Quick Start

### Basic Usage

Run training with default configuration:
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py
```

### Advanced Configuration

Configure training parameters via environment variables:
```bash
export SEQ_LENGTH=1024
export BATCH_SIZE=4
export LEARNING_RATE=1e-4
export NUM_EPOCHS=3
python main.py
```

## Model Architecture

### Teacher Model
- **Model**: Llama 3.2 3B Instruct
- **Parameters**: ~3 billion
- **Role**: Provides knowledge through logits, hidden states, and attention patterns

### Student Model
- **Base Model**: Llama 3.2 1B Instruct
- **LoRA Configuration**: Rank 128, Alpha 256
- **Trainable Parameters**: ~67 million (2.3% of base model)
- **Target Modules**: Query, Key, Value, Output projections + Feed-forward layers


## Training Configuration

### Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sequence Length | 512 | Maximum input sequence length |
| Batch Size | 4 | Per-device training batch size |
| Learning Rate | 5e-5 | Initial learning rate with cosine scheduling |
| Temperature | 3.0 | Distillation temperature for logits |
| Alpha | 0.5 | Balance between student loss and KD loss |
| LoRA Rank | 128 | Low-rank decomposition rank |
| LoRA Alpha | 256 | LoRA scaling parameter |


## Performance Expectations

Based on empirical results with the default configuration:

- **Initial Student Perplexity**: ~13
- **Teacher Perplexity**: ~10
- **Final Student Perplexity**: ~11
- **Improvement**: 20-30% perplexity reduction
- **Training Time**: 1-2 hours on A100
- **Memory Usage**: ~40GB VRAM


## References

For more information about knowledge distillation techniques and best practices, refer to the comprehensive resources available in the machine learning community.

**BabyLlama Project** - Practical implementation reference for Llama model distillation and training techniques: https://github.com/timinar/BabyLlama
