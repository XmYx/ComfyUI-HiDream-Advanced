"""Compatibility patches for older PyTorch versions"""
import torch
import torch.nn as nn

# Add RMSNorm if missing from PyTorch (needed for <2.1.0)
if not hasattr(nn, 'RMSNorm'):
    print("Adding RMSNorm implementation for compatibility with older PyTorch")
    
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        
        def forward(self, x):
            # Calculate RMS
            norm_x = torch.mean(x * x, dim=-1, keepdim=True)
            x_normed = x * torch.rsqrt(norm_x + self.eps)
            return self.weight * x_normed
    
    # Add to torch.nn for compatibility
    torch.nn.RMSNorm = RMSNorm
