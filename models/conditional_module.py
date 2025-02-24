# models/conditional_module.py
import torch
import torch.nn as nn

class ConditionalInitializer(nn.Module):
    def __init__(self, cond_dim, slot_dim):
        super().__init__()
        # A simple MLP to encode conditional inputs into the slot space.
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.ReLU(),
            nn.Linear(128, slot_dim)
        )

    def forward(self, cond_input):
        # cond_input: shape [B, cond_dim]
        return self.mlp(cond_input)
