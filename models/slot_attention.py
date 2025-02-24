# models/slot_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, hidden_dim=128, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_sigma)

        self.project_inputs = nn.Linear(dim, dim)
        self.project_slots = nn.Linear(dim, dim)
        self.project_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, slots_init=None):
        """
        Args:
            inputs: Tensor of shape [B, N, D] from the encoder.
            slots_init: Optional tensor of shape [B, num_slots, D] to initialize slots.
        Returns:
            slots: Updated slot representations.
        """
        B, N, D = inputs.shape
        inputs = self.norm_inputs(inputs)
        if slots_init is None:
            slots = self.slots_mu + self.slots_sigma * torch.randn(B, self.num_slots, self.dim, device=inputs.device)
        else:
            slots = slots_init

        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            k = self.project_inputs(inputs)
            v = self.project_v(inputs)
            q = self.project_slots(slots_norm)
            attn_logits = torch.einsum('bkd,bnd->bkn', q, k) / math.sqrt(D)
            attn = F.softmax(attn_logits, dim=1)
            updates = torch.einsum('bkn,bnd->bkd', attn, v)
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            ).reshape(B, self.num_slots, D)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        return slots
