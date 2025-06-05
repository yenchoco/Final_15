"""
Knowledge distillation trainer implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import logging
from transformers import Trainer
import numpy as np

logger = logging.getLogger(__name__)

class VocabAlignedDistillationTrainer(Trainer):
    """
    Knowledge distillation trainer with vocabulary alignment fixes
    """
    def __init__(self, *args, teacher_model=None, distill_config=None, **kwargs):
        self.teacher = teacher_model
        self.distill_config = distill_config
        
        super().__init__(*args, **kwargs)

        if self.teacher:
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

        self._projection_layers = {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        try:
            # Ensure input data types are correct
            if 'input_ids' in inputs:
                inputs['input_ids'] = inputs['input_ids'].to(model.device)
            if 'labels' in inputs:
                inputs['labels'] = inputs['labels'].to(model.device)

            # Student model forward pass
            outputs_student = model(**inputs, output_hidden_states=True, output_attentions=True)
            student_loss = outputs_student.loss

            if self.teacher is not None:
                with torch.no_grad():
                    # Ensure teacher and student models use the same vocabulary size
                    teacher_inputs = {k: v.to(self.teacher.device) for k, v in inputs.items()}
                    
                    # Handle vocabulary size mismatch
                    if 'input_ids' in teacher_inputs:
                        teacher_vocab_size = self.teacher.config.vocab_size
                        student_vocab_size = model.config.vocab_size
                        
                        if hasattr(model, 'base_model'):
                            student_vocab_size = model.base_model.config.vocab_size
                        
                        # Clip input_ids to the smaller vocabulary size
                        min_vocab_size = min(teacher_vocab_size, student_vocab_size)
                        teacher_inputs['input_ids'] = torch.clamp(teacher_inputs['input_ids'], 0, min_vocab_size - 1)
                        inputs['input_ids'] = torch.clamp(inputs['input_ids'], 0, min_vocab_size - 1)
                        
                        if 'labels' in teacher_inputs:
                            teacher_inputs['labels'] = torch.clamp(teacher_inputs['labels'], 0, min_vocab_size - 1)
                            inputs['labels'] = torch.clamp(inputs['labels'], 0, min_vocab_size - 1)
                    
                    outputs_teacher = self.teacher(**teacher_inputs, output_hidden_states=True, output_attentions=True)

                # 1. Vocabulary-aligned logits distillation
                logits_loss = self._compute_aligned_logits_distillation(
                    outputs_student.logits, 
                    outputs_teacher.logits,
                    inputs.get('labels', inputs['input_ids'])
                )

                # 2. Hidden states distillation
                hidden_loss = self._compute_hidden_distillation(
                    outputs_student.hidden_states,
                    outputs_teacher.hidden_states
                )

                # 3. Attention distillation - fixed version
                attention_loss = self._compute_attention_distillation_fixed(
                    outputs_student.attentions,
                    outputs_teacher.attentions,
                    inputs['input_ids']
                )

                # Combine all losses
                total_loss = (
                    self.distill_config['alpha'] * student_loss +
                    (1 - self.distill_config['alpha']) * (
                        self.distill_config['logits_weight'] * logits_loss +
                        self.distill_config['hidden_weight'] * hidden_loss +
                        self.distill_config['attention_weight'] * attention_loss
                    )
                )

                # Log detailed losses
                if self.state.global_step % self.args.logging_steps == 0:
                    wandb.log({
                        "train/student_loss": student_loss.item(),
                        "train/logits_distill_loss": logits_loss.item(),
                        "train/hidden_distill_loss": hidden_loss.item(),
                        "train/attention_distill_loss": attention_loss.item(),
                        "train/total_loss": total_loss.item(),
                        "train/step": self.state.global_step,
                    })

                    logger.info(
                        f"Step {self.state.global_step}: "
                        f"Student: {student_loss.item():.4f}, "
                        f"Logits KD: {logits_loss.item():.4f}, "
                        f"Hidden KD: {hidden_loss.item():.4f}, "
                        f"Attention KD: {attention_loss.item():.4f}, "
                        f"Total: {total_loss.item():.4f}"
                    )

                return (total_loss, outputs_student) if return_outputs else total_loss
            else:
                return (student_loss, outputs_student) if return_outputs else student_loss

        except Exception as e:
            logger.error(f"Error in compute_loss: {e}")
            raise
    
    def _compute_aligned_logits_distillation(self, student_logits, teacher_logits, labels):
        """
        Calculate vocabulary-aligned logits distillation loss
        """
        batch_size, seq_len, student_vocab_size = student_logits.shape
        teacher_vocab_size = teacher_logits.size(-1)
        
        # Ensure all tensors are on the same device and data type
        student_logits = student_logits.to(torch.float32)
        teacher_logits = teacher_logits.to(torch.float32)
        
        # Handle vocabulary size mismatch
        if student_vocab_size != teacher_vocab_size:
            min_vocab_size = min(student_vocab_size, teacher_vocab_size)
            
            # Clip to the same vocabulary size
            student_logits = student_logits[:, :, :min_vocab_size]
            teacher_logits = teacher_logits[:, :, :min_vocab_size]
            
            logger.debug(f"Aligned vocab sizes: student={student_logits.size(-1)}, teacher={teacher_logits.size(-1)}")
        
        student_logits_flat = student_logits.view(-1, student_logits.size(-1))
        teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))
        
        # Create mask to exclude padding tokens
        if hasattr(self, 'processing_class') and self.processing_class:
            pad_token_id = self.processing_class.pad_token_id
        else:
            pad_token_id = self.tokenizer.pad_token_id
            
        mask = (labels.view(-1) != pad_token_id) & (labels.view(-1) >= 0)
        
        if mask.sum() > 0:
            valid_student_logits = student_logits_flat[mask]
            valid_teacher_logits = teacher_logits_flat[mask]
            
            soft_targets = F.softmax(valid_teacher_logits / self.distill_config['temperature'], dim=-1)
            soft_prob = F.log_softmax(valid_student_logits / self.distill_config['temperature'], dim=-1)
            
            kl_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (self.distill_config['temperature'] ** 2)
            return kl_loss
        else:
            return torch.tensor(0.0, device=student_logits.device)
    
    def _compute_hidden_distillation(self, student_hidden, teacher_hidden):
        """Calculate hidden states distillation loss"""
        if not student_hidden or not teacher_hidden:
            return torch.tensor(0.0)
        
        student_layers = len(student_hidden) - 1
        teacher_layers = len(teacher_hidden) - 1
        
        layer_mapping = np.linspace(0, teacher_layers-1, student_layers, dtype=int)
        
        hidden_loss = 0.0
        count = 0
        
        for i, teacher_idx in enumerate(layer_mapping):
            student_h = student_hidden[i+1]
            teacher_h = teacher_hidden[teacher_idx+1]
            
            student_h = student_h.to(torch.float32)
            teacher_h = teacher_h.to(torch.float32)
            
            # Dimension alignment
            if student_h.size(-1) != teacher_h.size(-1):
                projection_key = f"hidden_{teacher_h.size(-1)}_{student_h.size(-1)}"
                
                if projection_key not in self._projection_layers:
                    self._projection_layers[projection_key] = nn.Linear(
                        teacher_h.size(-1), student_h.size(-1)
                    ).to(student_h.device).to(torch.float32)
                
                teacher_h = self._projection_layers[projection_key](teacher_h)
            
            loss = F.mse_loss(student_h, teacher_h.detach())
            hidden_loss += loss
            count += 1
        
        return hidden_loss / count if count > 0 else torch.tensor(0.0)

    
    def _compute_attention_distillation_fixed(self, student_attentions, teacher_attentions, input_ids):
        """
        Fixed version of attention distillation loss calculation
        """
        if not student_attentions or not teacher_attentions:
            return torch.tensor(0.0)
        
        num_layers_to_use = min(4, len(student_attentions), len(teacher_attentions))
        
        attention_loss = 0.0
        count = 0
        
        # Get actual sequence length (excluding padding)
        if hasattr(self, 'processing_class') and self.processing_class:
            pad_token_id = self.processing_class.pad_token_id
        else:
            pad_token_id = self.tokenizer.pad_token_id
        
        # Create attention mask
        attention_mask = (input_ids != pad_token_id).float()
        seq_len = attention_mask.sum(dim=1).max().int().item()
        
        for i in range(-num_layers_to_use, 0):
            try:
                student_att = student_attentions[i]
                teacher_att = teacher_attentions[i]
                
                # Ensure data types
                student_att = student_att.to(torch.float32)
                teacher_att = teacher_att.to(torch.float32)
                
                # Get dimension information
                batch_size = student_att.size(0)
                student_heads = student_att.size(1)
                teacher_heads = teacher_att.size(1)
                student_seq_len = student_att.size(2)
                teacher_seq_len = teacher_att.size(2)
                
                # Sequence length alignment
                min_seq_len = min(student_seq_len, teacher_seq_len, seq_len)
                student_att = student_att[:, :, :min_seq_len, :min_seq_len]
                teacher_att = teacher_att[:, :, :min_seq_len, :min_seq_len]
                
                # Attention head alignment
                if student_heads != teacher_heads:
                    if student_heads < teacher_heads:
                        # Teacher has more heads, need average pooling
                        heads_per_student = teacher_heads // student_heads
                        teacher_att = teacher_att.view(
                            batch_size, student_heads, heads_per_student,
                            min_seq_len, min_seq_len
                        ).mean(dim=2)
                    else:
                        # Student has more heads, repeat teacher attention
                        repeat_factor = student_heads // teacher_heads
                        teacher_att = teacher_att.repeat_interleave(repeat_factor, dim=1)
                        teacher_att = teacher_att[:, :student_heads, :, :]
                
                # Ensure dimensions match completely
                if student_att.shape != teacher_att.shape:
                    logger.warning(f"Attention shape mismatch after alignment: {student_att.shape} vs {teacher_att.shape}")
                    continue
                
                # Apply attention mask
                mask = attention_mask[:, :min_seq_len].unsqueeze(1).unsqueeze(1)
                mask = mask * mask.transpose(-1, -2)
                
                # Only calculate loss at valid positions
                valid_positions = mask.bool()
                
                if valid_positions.sum() > 0:
                    student_att_masked = student_att[valid_positions]
                    teacher_att_masked = teacher_att[valid_positions]
                    
                    # Calculate KL divergence
                    loss = F.kl_div(
                        F.log_softmax(student_att_masked, dim=-1),
                        F.softmax(teacher_att_masked, dim=-1),
                        reduction='batchmean'
                    )
                    
                    attention_loss += loss
                    count += 1
                
            except Exception as e:
                logger.warning(f"Error in attention distillation layer {i}: {e}")
                continue
        
        return attention_loss / count if count > 0 else torch.tensor(0.0)