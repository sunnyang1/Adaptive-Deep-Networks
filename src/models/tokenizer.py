"""
Tokenizer for Adaptive Deep Networks

Uses HuggingFace transformers tokenizer with custom vocabulary handling.
"""

from typing import List, Union
import torch
from transformers import GPT2Tokenizer


class SimpleTokenizer:
    """
    Simple tokenizer wrapper for testing and training.
    
    Uses GPT2Tokenizer as base and provides encode/decode interface
    compatible with the model.
    """
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        
        # Use GPT2Tokenizer as base
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            # Resize vocabulary if needed
            if vocab_size != self.tokenizer.vocab_size:
                # For simplicity, we'll just use a subset or pad
                pass
        except (OSError, ValueError, RuntimeError) as e:
            # Fallback to character-level tokenizer for testing
            self.tokenizer = None
            self.char_vocab = {chr(i): i for i in range(vocab_size)}
            self.inv_vocab = {i: chr(i) for i in range(vocab_size)}
    
    def encode(
        self,
        text: str,
        return_tensors: str = None,
        max_length: int = None
    ) -> Union[List[int], torch.Tensor]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            return_tensors: If 'pt', return PyTorch tensor
            max_length: Maximum sequence length
        
        Returns:
            Token IDs
        """
        if self.tokenizer is not None:
            # Use GPT2Tokenizer
            if max_length:
                tokens = self.tokenizer.encode(
                    text,
                    max_length=max_length,
                    truncation=True
                )
            else:
                tokens = self.tokenizer.encode(text)
        else:
            # Character-level fallback
            tokens = [self.char_vocab.get(c, 0) for c in text[:max_length]]
        
        if return_tensors == 'pt':
            return torch.tensor([tokens])
        return tokens
    
    def decode(self, tokens: Union[List[int], torch.Tensor]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            tokens: Token IDs
        
        Returns:
            Decoded text
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        if self.tokenizer is not None:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        else:
            return ''.join(self.inv_vocab.get(t, '') for t in tokens)
    
    def batch_encode(
        self,
        texts: List[str],
        padding: bool = True,
        max_length: int = None
    ) -> torch.Tensor:
        """
        Encode batch of texts.
        
        Args:
            texts: List of input texts
            padding: Whether to pad to max length
            max_length: Maximum sequence length
        
        Returns:
            Batched tensor of token IDs
        """
        encoded = [self.encode(t, max_length=max_length) for t in texts]
        
        if padding:
            max_len = max(len(e) for e in encoded)
            encoded = [e + [0] * (max_len - len(e)) for e in encoded]
        
        return torch.tensor(encoded)
    
    def __len__(self):
        return self.vocab_size


def create_tokenizer(vocab_size: int = 32000) -> SimpleTokenizer:
    """Factory function to create tokenizer."""
    return SimpleTokenizer(vocab_size)
