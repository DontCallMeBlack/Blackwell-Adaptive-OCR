import torch

def apply_2_4_sparsity(tensor: torch.Tensor) -> torch.Tensor:
    """
    Simulates NVIDIA Blackwell 2:4 structured sparsity.
    In every block of 4 elements, only the 2 with the highest magnitude remain.
    """
    if tensor.shape[-1] % 4 != 0:
        return tensor  # Skip if not divisible by 4
    
    original_shape = tensor.shape
    reshaped = tensor.view(-1, 4)
    
    # Identify 2 smallest values
    _, indices = torch.topk(reshaped.abs(), 2, dim=1, largest=False)
    
    # Create mask
    mask = torch.ones_like(reshaped)
    mask.scatter_(1, indices, 0)
    
    return (reshaped * mask).view(original_shape)

def simulate_quantization(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric min-max quantization simulation."""
    q_max = 2**(bits - 1) - 1
    q_min = -2**(bits - 1)
    
    scale = tensor.abs().max() / q_max
    if scale == 0: return tensor
    
    return torch.clamp(torch.round(tensor / scale), q_min, q_max) * scale