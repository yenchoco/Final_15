"""
Training configuration management
"""
from dataclasses import dataclass
import os

@dataclass
class LoRAConfig:
    """LoRA specific configuration"""
    r: int = 128
    alpha: int = 256
    dropout: float = 0.1
    target_modules: list = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass
class DistillationConfig:
    """Knowledge distillation configuration"""
    logits_weight: float = 0.6
    hidden_weight: float = 0.3
    attention_weight: float = 0.1
    temperature: float = 3.0
    alpha: float = 0.5

@dataclass
class TrainingConfig:
    """Main training configuration"""
    # Model parameters
    seq_length: int = int(os.getenv('SEQ_LENGTH', 512))
    batch_size: int = int(os.getenv('BATCH_SIZE', 4))
    learning_rate: float = float(os.getenv('LEARNING_RATE', 5e-5))
    num_epochs: int = int(os.getenv('NUM_EPOCHS', 1))
    
    # Training parameters
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 300
    logging_steps: int = 25
    save_steps: int = 500
    eval_steps: int = 250
    weight_decay: float = 0.01
    warmup_ratio: float = 0.15
    
    # Data parameters
    max_train_samples: int = 8000
    max_eval_samples: int = 1500
    
    # Nested configurations
    lora: LoRAConfig = None
    distillation: DistillationConfig = None
    
    def __post_init__(self):
        if self.lora is None:
            self.lora = LoRAConfig()
        if self.distillation is None:
            self.distillation = DistillationConfig()
    
    def validate(self):
        """Validate configuration parameters"""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.seq_length <= 0:
            raise ValueError("Sequence length must be positive")

# Create global configuration instance
training_config = TrainingConfig()