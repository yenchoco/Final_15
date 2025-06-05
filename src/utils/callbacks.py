"""
Training callbacks
"""
import numpy as np
import logging
import wandb
from transformers import TrainerCallback
from .evaluation import evaluate_ppl

logger = logging.getLogger(__name__)

class LoRAPerplexityCallback(TrainerCallback):
    def __init__(self, tokenizer, device="cuda:0", eval_steps=500):
        self.tokenizer = tokenizer
        self.device = device
        self.eval_steps = eval_steps
        self.best_ppl = float('inf')

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_steps == 0:
            logger.info(f"Computing perplexity at step {state.global_step}...")

            try:
                ppl = evaluate_ppl(model, self.tokenizer, self.device)
                lora_stats = self._compute_lora_stats(model)

                wandb.log({
                    "eval/perplexity": ppl,
                    "eval/step": state.global_step,
                    **lora_stats
                })

                logger.info(f"Step {state.global_step} - Perplexity: {ppl:.4f}")

                if ppl < self.best_ppl:
                    self.best_ppl = ppl
                    wandb.log({"eval/best_perplexity": self.best_ppl})
                    logger.info(f"New best perplexity: {ppl:.4f}")

            except Exception as e:
                logger.error(f"Error computing perplexity: {e}")

    def _compute_lora_stats(self, model):
        lora_params = 0
        total_params = 0
        lora_norms = []
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            if 'lora_' in name:
                lora_params += param.numel()
                lora_norms.append(param.data.norm().item())
        
        return {
            "lora/trainable_params": lora_params,
            "lora/total_params": total_params,
            "lora/trainable_ratio": lora_params / total_params * 100,
            "lora/avg_norm": np.mean(lora_norms) if lora_norms else 0,
            "lora/max_norm": np.max(lora_norms) if lora_norms else 0,
        }