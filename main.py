"""
Main entry point for training
"""
import torch
import wandb
import logging
import sys
import warnings

from config import training_config, model_config
from src.training.train import run_training

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

def main():
    """Main function"""
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Initialize wandb
    wandb.init(
        project=model_config.project_name,
        config={
            **training_config.__dict__,
            **model_config.__dict__,
            "method": "KD+LoRA+VocabAligned+AttentionFixed"
        }
    )
    
    try:
        # Run training
        trainer, student_model, teacher_model, teacher_tokenizer = run_training()
        logger.info("Training completed successfully!")

        # Save model
        print("Saving vocab-aligned LoRA model...")
        trainer.save_model()
        teacher_tokenizer.save_pretrained(model_config.output_dir)
        student_model.save_pretrained(model_config.output_dir)

        # Upload to Hugging Face Hub
        print("Uploading to Hugging Face Hub...")
        trainer.model.push_to_hub(model_config.hub_model_name)
        teacher_tokenizer.push_to_hub(model_config.hub_model_name)
        print("Successfully uploaded to Hugging Face Hub!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()