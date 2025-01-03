from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()


# Reset positions for packed samples
# For example
# Input: attention_mask = torch.tensor([[1, 1, 1, 2, 2, 2, 3, 3, 0]])
# Output: position_ids  = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 0]])
def reset_position_ids(attention_mask):
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
    for i in range(attention_mask.size(0)):
        mask = attention_mask[i]
        seq_num = mask.max().item()
        for index in range(1, seq_num + 1):
            sample_mask = mask == index
            sample_length = sample_mask.sum().item()
            position_ids[i, sample_mask] = torch.arange(sample_length, device=mask.device)
    return position_ids


def unpacking_samples(values: torch.Tensor, packed_seqlens: list[int]):
    values = values.squeeze(0)
    unpacked_values = []
    offset = 0
    for seqlen in packed_seqlens:
        unpacked_values.append(values[offset : offset + seqlen])
        offset += seqlen
    return unpacked_values