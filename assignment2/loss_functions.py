"""Utility functions for student-implemented loss computations.

The training entry point expects a callable named `compute_loss_from_logits`.
Students should implement the function so that it takes model logits and
ground truth labels and returns a scalar loss tensor.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from transformers.trainer_pt_utils import LabelSmoother

from transformers.modeling_outputs import CausalLMOutputWithPast


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def compute_loss_from_logits(
    outputs: CausalLMOutputWithPast,
    labels: Optional[torch.Tensor],
    num_items_in_batch: int,
) -> torch.Tensor:
    """Compute the token-level cross-entropy loss for language modeling.

    Args:
        logits: Float tensor with shape [batch_size, seq_len, vocab_size].
        labels: Long tensor with shape [batch_size, seq_len].
        ignore_index: Label id that should be ignored when computing the loss. The
            trainer passes HuggingFace's default ignore index (-100).

    Returns:
        Scalar tensor representing the mean loss over non-ignored tokens.

    Students should implement this function by computing the cross-entropy loss
    from the raw logits. You may not call `torch.nn.CrossEntropyLoss`; instead,
    derive the loss explicitly using a log-softmax over the vocabulary dimension.
    """

    # raise NotImplementedError("Implement token-level cross-entropy using the logits.")
    logits = outputs.logits
    return cross_entropy_loss(logits, labels, num_items_in_batch=num_items_in_batch)


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_items_in_batch: int,
) -> torch.Tensor:
    """
    Compute the token-level cross-entropy loss for language modeling.

    Args:
        logits: Float tensor with shape [batch_size, seq_len, vocab_size].
        labels: Long tensor with shape [batch_size, seq_len].
        num_items_in_batch: Number of valid items in batch for normalization.

    Returns:
        Scalar tensor representing the mean loss over non-ignored tokens.
    """

    #casual shift 
    logits = logits[:, :-1, :].contiguous()  # shape: [batch_size, seq_len-1, vocab_size]
    labels = labels[:, 1:].contiguous()      # shape: [batch_size, seq_len-1]

    # exp_logits = torch.exp(logits) 
    # sum_exp = torch.sum(exp_logits, dim=-1, keepdim=True) # shape: [batch_size, seq_len-1, 1]
    # log_softmax_result = torch.log(exp_logits / sum_exp) # shape: [batch_size, seq_len-1, vocab_size]
    log_softmax_result = F.log_softmax(logits, dim=-1) # shape: [batch_size, seq_len-1, vocab_size]
    
    safe_labels = labels.clone()
    safe_labels[labels == IGNORE_TOKEN_ID] = 0  # to avoid indexing errors by setting ignored labels to a valid index

    #ce = -F.one_hot(labels, num_classes=logits.size(-1)) * log_softmax_result # shape: [batch_size, seq_len, vocab_size]
    ce = -torch.gather(log_softmax_result, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1) # Another way, # shape: [batch_size, seq_len]

    mask = (labels != IGNORE_TOKEN_ID) # shape: [batch_size, seq_len-1]
    ce = ce * mask # shape: [batch_size, seq_len-1]

    if num_items_in_batch != 0:
        avg_loss = torch.sum(ce) / num_items_in_batch # shape: [batch_size, seq_len-1]
        return avg_loss
    else:
        return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)


    #raise NotImplementedError("Implement token-level cross-entropy using the logits.")
