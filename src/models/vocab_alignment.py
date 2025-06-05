"""
Vocabulary alignment utilities
"""
import logging

logger = logging.getLogger(__name__)

def align_vocabularies(teacher_model, student_model, teacher_tokenizer, student_tokenizer):
    """
    Resolve vocabulary size mismatch issues in Llama 3.2 model series
    """
    teacher_vocab_size = teacher_model.config.vocab_size
    student_vocab_size = student_model.config.vocab_size
    
    logger.info(f"Teacher vocab size: {teacher_vocab_size}")
    logger.info(f"Student vocab size: {student_vocab_size}")
    
    # Check if vocabulary sizes match
    if teacher_vocab_size != student_vocab_size:
        logger.warning(f"Vocabulary size mismatch detected: {teacher_vocab_size} vs {student_vocab_size}")
        
        # Use the smaller vocabulary size as a unified standard
        target_vocab_size = min(teacher_vocab_size, student_vocab_size)
        logger.info(f"Aligning to smaller vocabulary size: {target_vocab_size}")
        
        # Adjust teacher model vocabulary
        if teacher_vocab_size > target_vocab_size:
            teacher_model.resize_token_embeddings(target_vocab_size)
            logger.info(f"Resized teacher model vocabulary to {target_vocab_size}")
        
        # Adjust student model vocabulary
        if student_vocab_size > target_vocab_size:
            student_model.resize_token_embeddings(target_vocab_size)
            logger.info(f"Resized student model vocabulary to {target_vocab_size}")
        
        # Update tokenizer vocabulary size (if needed)
        teacher_tokenizer.model_max_length = min(teacher_tokenizer.model_max_length, target_vocab_size)
        student_tokenizer.model_max_length = min(student_tokenizer.model_max_length, target_vocab_size)
        
        return target_vocab_size
    
    return teacher_vocab_size