"""
Main training logic
"""
import torch
import json
import logging
import wandb
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    DataCollatorForLanguageModeling,
    TrainingArguments, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType

from config import training_config, model_config
from src.models.vocab_alignment import align_vocabularies
from src.models.distillation_trainer import VocabAlignedDistillationTrainer
from src.data.dataset import WikiText2Dataset
from src.utils.callbacks import LoRAPerplexityCallback
from src.utils.evaluation import evaluate_ppl

logger = logging.getLogger(__name__)

def setup_models():
    """Setup teacher and student models"""
    # Load teacher model
    teacher_tokenizer = AutoTokenizer.from_pretrained(model_config.teacher_model_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_config.teacher_model_name,
        torch_dtype=getattr(torch, model_config.torch_dtype),
        device_map=model_config.device_map,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation
    )
    
    # Load student model
    student_tokenizer = AutoTokenizer.from_pretrained(model_config.student_model_name)
    student_base_model = AutoModelForCausalLM.from_pretrained(
        model_config.student_model_name,
        torch_dtype=getattr(torch, model_config.torch_dtype),
        device_map=model_config.device_map,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation
    )
    
    # Set pad tokens
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id
    
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
        student_tokenizer.pad_token_id = student_tokenizer.eos_token_id
    
    return teacher_model, teacher_tokenizer, student_base_model, student_tokenizer

def setup_lora_model(student_base_model):
    """Setup LoRA configuration and model"""
    lora_config = LoraConfig(
        r=training_config.lora.r,
        lora_alpha=training_config.lora.alpha,
        target_modules=training_config.lora.target_modules,
        lora_dropout=training_config.lora.dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )
    
    student_model = get_peft_model(student_base_model, lora_config)
    return student_model

def run_training():
    """Main training function"""
    # Validate configuration
    training_config.validate()
    
    # Setup models
    teacher_model, teacher_tokenizer, student_base_model, student_tokenizer = setup_models()
    
    # Align vocabularies
    aligned_vocab_size = align_vocabularies(
        teacher_model, student_base_model, 
        teacher_tokenizer, student_tokenizer
    )
    
    # Setup LoRA
    student_model = setup_lora_model(student_base_model)
    student_model.print_trainable_parameters()
    
    # Setup datasets
    train_dataset = WikiText2Dataset(
        split='train',
        seq_length=training_config.seq_length,
        tokenizer=teacher_tokenizer,
        max_samples=training_config.max_train_samples
    )
    
    eval_dataset = WikiText2Dataset(
        split='validation',
        seq_length=training_config.seq_length,
        tokenizer=teacher_tokenizer,
        max_samples=training_config.max_eval_samples
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=model_config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        warmup_steps=training_config.warmup_steps,
        logging_steps=training_config.logging_steps,
        save_steps=training_config.save_steps,
        eval_steps=training_config.eval_steps,
        evaluation_strategy="steps",
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        fp16=True,
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        run_name=model_config.run_name,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        warmup_ratio=training_config.warmup_ratio,
        save_strategy="steps",
        save_only_model=True,
        dataloader_num_workers=2,
        group_by_length=True,
    )
    
    # Setup trainer
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=teacher_tokenizer,
        mlm=False,
    )
    
    trainer = VocabAlignedDistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=teacher_tokenizer,
        distill_config=training_config.distillation.__dict__,
    )
    
    # Add callbacks
    ppl_callback = LoRAPerplexityCallback(
        tokenizer=teacher_tokenizer,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        eval_steps=training_config.eval_steps
    )
    
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=4,
        early_stopping_threshold=0.001
    )
    
    trainer.add_callback(ppl_callback)
    trainer.add_callback(early_stopping_callback)
    
    # Start training
    trainer.train()
    
    # Save model
    trainer.save_model()
    teacher_tokenizer.save_pretrained(model_config.output_dir)
    student_model.save_pretrained(model_config.output_dir)
    
    # Push to hub
    trainer.model.push_to_hub(model_config.hub_model_name)
    teacher_tokenizer.push_to_hub(model_config.hub_model_name)
    
    return trainer, student_model, teacher_model, teacher_tokenizer