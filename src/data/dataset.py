"""
Dataset implementations
"""
import logging
from datasets import load_dataset
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class WikiText2Dataset(Dataset):
    def __init__(self, split='train', seq_length=512, tokenizer=None, max_samples=None):
        self.seq_length = seq_length
        self.tokenizer = tokenizer

        logger.info(f"Loading WikiText-2 {split} dataset...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)

        self.examples = []
        for example in dataset:
            text = example['text'].strip()
            if text and len(text) > 200 and len(text) < 2000:
                if not text.startswith('=') and not text.startswith('-') and not text.startswith('*'):
                    self.examples.append(text)

                if max_samples and len(self.examples) >= max_samples:
                    break

        logger.info(f"Loaded {len(self.examples)} text examples for {split}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.seq_length + 1,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()

        return {
            'input_ids': input_ids[:-1],
            'labels': input_ids[1:]
        }