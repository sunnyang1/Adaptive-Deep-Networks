"""
Data Loading Utilities

Shared dataset and dataloader implementations.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Iterator
import itertools


class HuggingFaceDataset(Dataset):
    """
    Dataset backed by HuggingFace datasets library.
    
    Supports streaming mode for large datasets without local storage.
    """
    
    def __init__(
        self,
        dataset_name: str = "openwebtext",
        dataset_config: Optional[str] = None,
        split: str = "train",
        seq_len: int = 512,
        tokenizer=None,
        max_samples: int = 100000,
        streaming: bool = True,
        text_column: str = "text",
    ):
        """
        Args:
            dataset_name: HuggingFace dataset name (e.g., "openwebtext", "HuggingFaceFW/fineweb-edu")
            dataset_config: Dataset configuration name
            split: Dataset split ("train", "validation", "test")
            seq_len: Sequence length for training
            tokenizer: Tokenizer instance (with encode method)
            max_samples: Maximum number of samples (for non-streaming mode)
            streaming: Whether to use streaming mode (no local download)
            text_column: Column name containing text
        """
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.text_column = text_column
        
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        print(f"Loading dataset: {dataset_name} ({dataset_config or 'default'}) [{split}]")
        print(f"Streaming: {streaming}, Max samples: {max_samples}")
        
        self.streaming = streaming
        self.dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=streaming,
        )
        
        # For non-streaming, limit samples
        if not streaming and max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        # Pre-tokenize for faster access (if not streaming and small enough)
        self.pre_tokenized = not streaming and max_samples and max_samples <= 10000
        if self.pre_tokenized:
            print("Pre-tokenizing dataset...")
            self.tokenized_data = self._pre_tokenize()
        else:
            self.tokenized_data = None
        
        # For streaming mode, cache first N items in memory
        self.streaming_buffer = None
        if streaming and max_samples:
            print(f"Caching first {max_samples} samples from streaming dataset...")
            self.streaming_buffer = list(itertools.islice(self.dataset, max_samples))
            print(f"Cached {len(self.streaming_buffer)} samples")
    
    def _pre_tokenize(self):
        """Pre-tokenize entire dataset for faster access."""
        data = []
        for item in self.dataset:
            text = item.get(self.text_column, "")
            if not text:
                continue
            tokens = self._tokenize_text(text)
            if len(tokens) >= self.seq_len + 1:
                data.append(tokens)
        return data
    
    def _tokenize_text(self, text: str) -> list:
        """Tokenize text to token IDs."""
        if self.tokenizer is not None:
            # Use tokenizer's encode method (handles both old and new interface)
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            # Handle both list and tensor returns
            if isinstance(encoded, torch.Tensor):
                return encoded.tolist()
            return encoded
        # Fallback: simple byte encoding
        return [b % 32000 for b in text.encode('utf-8', errors='ignore')]
    
    def __len__(self) -> int:
        if self.pre_tokenized:
            return len(self.tokenized_data)
        if self.streaming_buffer is not None:
            return len(self.streaming_buffer)
        # For streaming datasets without buffer, return a large number
        return 1000000
    
    def __getitem__(self, idx: int) -> dict:
        if self.pre_tokenized:
            tokens = self.tokenized_data[idx]
            input_ids = tokens[:self.seq_len]
            labels = tokens[1:self.seq_len + 1]
        elif self.streaming_buffer is not None:
            # Use cached streaming data
            item = self.streaming_buffer[idx]
            # Handle different item formats
            if hasattr(item, 'get'):
                # Dictionary-like object
                text = item.get(self.text_column, "")
            elif hasattr(item, self.text_column):
                # Object with attribute access (e.g., IterableColumn)
                text = getattr(item, self.text_column, "")
            else:
                # Fallback: try direct access
                try:
                    text = item[self.text_column] if isinstance(item, dict) else str(item)
                except (TypeError, KeyError):
                    text = ""
            
            tokens = self._tokenize_text(text)
            
            # Create sliding window if text is long enough
            if len(tokens) >= self.seq_len + 1:
                input_ids = tokens[:self.seq_len]
                labels = tokens[1:self.seq_len + 1]
            else:
                # Pad short sequences
                input_ids = tokens[:-1] + [0] * (self.seq_len - len(tokens) + 1)
                labels = tokens[1:] + [0] * (self.seq_len - len(tokens) + 1)
        else:
            # Pure streaming mode without buffer - not supported for random access
            raise RuntimeError(
                "Streaming mode without buffer does not support random access. "
                "Please set max_samples to cache data in memory, or use non-streaming mode."
            )
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


class DummyDataset(Dataset):
    """
    Dummy dataset for testing.
    
    Generates random data of specified shape.
    """
    
    def __init__(
        self,
        size: int = 1000,
        seq_len: int = 512,
        vocab_size: int = 32000,
        seed: int = 42
    ):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.seed = seed
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> dict:
        # Deterministic per-sample generator for reproducibility
        g = torch.Generator().manual_seed(self.seed + idx)
        
        # Generate random sequence
        input_ids = torch.randint(
            0, self.vocab_size,
            (self.seq_len,),
            generator=g
        )
        
        # Shift for labels (next token prediction)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = 0  # End token
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }


class StreamingTextDataset(Dataset):
    """
    Streaming dataset for large text corpora.
    
    Loads data on-the-fly to avoid memory issues.
    """
    
    def __init__(
        self,
        data_iterator: Iterator,
        seq_len: int = 2048,
        max_samples: Optional[int] = None
    ):
        self.data_iterator = data_iterator
        self.seq_len = seq_len
        self.max_samples = max_samples
        self.buffer = []
        self._fill_buffer()
    
    def _fill_buffer(self):
        """Fill buffer from iterator."""
        try:
            while len(self.buffer) < (self.max_samples or 1000):
                item = next(self.data_iterator)
                self.buffer.append(item)
        except StopIteration:
            pass
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def __getitem__(self, idx: int) -> dict:
        if idx >= len(self.buffer):
            raise IndexError(f"Index {idx} out of range")
        
        text = self.buffer[idx]
        
        # Tokenize (simplified - should use actual tokenizer)
        tokens = text.split()[:self.seq_len]
        input_ids = torch.tensor([hash(t) % 32000 for t in tokens], dtype=torch.long)
        
        # Pad to seq_len
        if len(input_ids) < self.seq_len:
            input_ids = torch.cat([
                input_ids,
                torch.zeros(self.seq_len - len(input_ids), dtype=torch.long)
            ])
        
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create dataloader with standard settings.
    
    Args:
        dataset: Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Pin memory for CUDA
    
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
