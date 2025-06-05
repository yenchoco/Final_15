"""
Model evaluation utilities
"""
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm.auto import tqdm
import logging

logger = logging.getLogger(__name__)

def evaluate_ppl(model, tokenizer, device="cuda:0", max_length=2048):
    """Evaluate model perplexity"""
    model.eval()
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    test_text = "\n\n".join([text for text in test_dataset["text"] if text.strip()])
    test_enc = tokenizer(test_text, return_tensors="pt")
    test_enc = test_enc.input_ids.to(device)

    seqlen = min(max_length, 2048)
    nsamples = test_enc.numel() // seqlen

    if nsamples == 0:
        return float('inf')

    nlls = []

    for i in tqdm(range(nsamples), desc="Evaluating PPL"):
        batch = test_enc[:, (i * seqlen):((i + 1) * seqlen)]

        with torch.no_grad():
            try:
                outputs = model(batch)
                lm_logits = outputs.logits

                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()

                shift_logits = shift_logits.to(torch.float32)
                
                loss_fct = nn.CrossEntropyLoss(reduction='mean')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                               shift_labels.view(-1))

                neg_log_likelihood = loss.float() * seqlen
                nlls.append(neg_log_likelihood)

            except Exception as e:
                logger.warning(f"Error processing batch {i}: {e}")
                continue

    if not nlls:
        return float('inf')

    ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
    model.train()

    return ppl.item()