"""
Model configuration management
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Model paths and settings"""
    teacher_model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    student_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    
    # Model settings
    torch_dtype: str = "float16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    attn_implementation: str = "eager"
    
    # Output settings
    output_dir: str = "./llama32-wikitext2-kd-lora-vocab-aligned-fixed"
    hub_model_name: str = "your-HugginFace/kd-lora-5e5-lora128-ep1"
    
    # Wandb settings
    project_name: str = "llama32-wikitext2-kd-lora-vocab-aligned"
    run_name: str = "llama32-kd-lora-vocab-aligned-fixed"

model_config = ModelConfig()