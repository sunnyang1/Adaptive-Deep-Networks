"""
Tokenizer for Adaptive Deep Networks

Supports multiple tokenizers including GPT-2 and Llama-2.
"""

from typing import List, Union, Optional
import torch
from transformers import PreTrainedTokenizer


class TokenizerWrapper:
    """
    Wrapper for HuggingFace tokenizers with consistent interface.
    
    Supports:
    - GPT-2 (gpt2)
    - Llama-2 (meta-llama/Llama-2-7b-hf, requires auth)
    - Llama-3 (meta-llama/Meta-Llama-3-8B)
    - Custom tokenizers
    """
    
    # Tokenizer vocab sizes
    VOCAB_SIZES = {
        'gpt2': 50257,
        'meta-llama/Llama-2-7b-hf': 32000,
        'meta-llama/Llama-2-13b-hf': 32000,
        'meta-llama/Llama-2-70b-hf': 32000,
        'meta-llama/Meta-Llama-3-8B': 128256,
        'meta-llama/Meta-Llama-3-70B': 128256,
    }
    
    def __init__(
        self,
        tokenizer_name: str = 'gpt2',
        vocab_size: Optional[int] = None,
        use_fast: bool = True,
        trust_remote_code: bool = False,
        token: Optional[str] = None,  # For HuggingFace auth (needed for Llama-2)
    ):
        """
        Initialize tokenizer.
        
        Args:
            tokenizer_name: HuggingFace tokenizer name or path
            vocab_size: Override vocab size (if None, use tokenizer's native size)
            use_fast: Use fast tokenizer implementation
            trust_remote_code: Trust remote code (needed for some custom tokenizers)
            token: HuggingFace API token (needed for gated models like Llama-2)
        """
        self.tokenizer_name = tokenizer_name
        
        try:
            from transformers import AutoTokenizer
            
            # Load the tokenizer
            print(f"Loading tokenizer: {tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                use_fast=use_fast,
                trust_remote_code=trust_remote_code,
                token=token,
            )
            
            # Set pad token if not set (Llama-2 doesn't have pad token by default)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Determine vocab size
            if vocab_size is not None:
                self.vocab_size = vocab_size
                if vocab_size != len(self.tokenizer):
                    print(f"Warning: Requested vocab_size={vocab_size} but tokenizer has {len(self.tokenizer)} tokens")
            else:
                self.vocab_size = len(self.tokenizer)
            
            print(f"Tokenizer loaded: vocab_size={self.vocab_size}, pad_token={self.tokenizer.pad_token}")
            
        except Exception as e:
            print(f"Failed to load tokenizer '{tokenizer_name}': {e}")
            print("Falling back to GPT2 tokenizer...")
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.vocab_size = vocab_size or len(self.tokenizer)
    
    def encode(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        max_length: Optional[int] = None,
        truncation: bool = True,
        add_special_tokens: bool = True,
    ) -> Union[List[int], torch.Tensor]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            return_tensors: If 'pt', return PyTorch tensor
            max_length: Maximum sequence length
            truncation: Whether to truncate
            add_special_tokens: Whether to add special tokens (BOS/EOS)
        
        Returns:
            Token IDs
        """
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            truncation=truncation,
        )
        
        if return_tensors == 'pt':
            return torch.tensor([encoded])
        return encoded
    
    def decode(
        self,
        tokens: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            tokens: Token IDs
            skip_special_tokens: Whether to skip special tokens in output
        
        Returns:
            Decoded text
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
    
    def batch_encode(
        self,
        texts: List[str],
        padding: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        return_tensors: str = 'pt',
    ) -> torch.Tensor:
        """
        Encode batch of texts.
        
        Args:
            texts: List of input texts
            padding: Whether to pad to max length
            max_length: Maximum sequence length
            truncation: Whether to truncate
            return_tensors: Return format ('pt' for PyTorch)
        
        Returns:
            Batched tensor of token IDs
        """
        encoded = self.tokenizer(
            texts,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded['input_ids']
    
    def __len__(self):
        return self.vocab_size
    
    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id
    
    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id
    
    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id


# Backwards compatibility
class SimpleTokenizer(TokenizerWrapper):
    """Legacy wrapper for backwards compatibility."""
    
    def __init__(self, vocab_size: int = 32000):
        super().__init__(tokenizer_name='gpt2', vocab_size=vocab_size)


def create_tokenizer(
    tokenizer_name: str = 'gpt2',
    vocab_size: Optional[int] = None,
    token: Optional[str] = None,
) -> TokenizerWrapper:
    """
    Factory function to create tokenizer.
    
    Args:
        tokenizer_name: HuggingFace tokenizer name
        vocab_size: Override vocab size
        token: HuggingFace API token (for gated models like Llama-2)
    
    Returns:
        TokenizerWrapper instance
    
    Examples:
        >>> # GPT-2 tokenizer
        >>> tokenizer = create_tokenizer('gpt2')
        
        >>> # Llama-2 tokenizer (requires HuggingFace auth)
        >>> tokenizer = create_tokenizer('meta-llama/Llama-2-7b-hf', token='hf_xxx')
        
        >>> # Llama-3 tokenizer
        >>> tokenizer = create_tokenizer('meta-llama/Meta-Llama-3-8B', token='hf_xxx')
    """
    return TokenizerWrapper(
        tokenizer_name=tokenizer_name,
        vocab_size=vocab_size,
        token=token,
    )


def get_tokenizer_for_model(model_name: str) -> str:
    """
    Get recommended tokenizer for a given model architecture.
    
    Args:
        model_name: Model size or architecture name
    
    Returns:
        Recommended tokenizer name
    """
    # For small model (1.1B), we can use GPT-2 or Llama-2 tokenizer
    # Llama-2 tokenizer is generally better for language modeling
    tokenizers = {
        'gpt2': 'gpt2',
        'llama2': 'meta-llama/Llama-2-7b-hf',
        'llama3': 'meta-llama/Meta-Llama-3-8B',
        't4': 'gpt2',
        'small': 'gpt2',  # Default for small model
        'medium': 'meta-llama/Llama-2-7b-hf',  # Recommend Llama-2 for medium+
        'large': 'meta-llama/Llama-2-7b-hf',
    }
    return tokenizers.get(model_name.lower(), 'gpt2')
