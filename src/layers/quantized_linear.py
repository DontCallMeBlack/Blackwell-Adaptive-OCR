import torch
import torch.nn as nn
from src.utils.hardware_utils import apply_2_4_sparsity, simulate_quantization

class BlackwellLinear(nn.Module):
    def __init__(self, in_features, out_features, precision=8, use_sparsity=True):
        super().__init__()
        self.precision = precision
        self.use_sparsity = use_sparsity
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        w = self.weight
        
        if self.use_sparsity:
            w = apply_2_4_sparsity(w)
            
        w = simulate_quantization(w, self.precision)
        
        return torch.nn.functional.linear(x, w, self.bias)