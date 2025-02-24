# models/slot_predictor.py
import torch
import torch.nn as nn

class SlotPredictor(nn.Module):
    def __init__(self, slot_dim, hidden_dim=128, num_heads=4):
        super().__init__()
        # Use multihead self-attention to model interactions among slots
        self.self_attn = nn.MultiheadAttention(embed_dim=slot_dim, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim)
        )
        self.norm1 = nn.LayerNorm(slot_dim)
        self.norm2 = nn.LayerNorm(slot_dim)
        
    def forward(self, slots):
        # slots: [B, num_slots, slot_dim]
        attn_out, _ = self.self_attn(slots, slots, slots)
        slots = self.norm1(slots + attn_out)
        mlp_out = self.mlp(slots)
        slots = self.norm2(slots + mlp_out)
        return slots
