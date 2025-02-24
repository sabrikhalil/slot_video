# models/autoencoder.py
import torch
import torch.nn as nn
from models.encoder import Encoder
from models.slot_attention import SlotAttention
from models.decoder import SpatialBroadcastDecoder
from models.slot_predictor import SlotPredictor

class ConditionalVideoSlotAutoencoder(nn.Module):
    def __init__(self, num_slots=10, slot_dim=64, cond_dim=10):
        """
        Args:
            num_slots (int): Number of slots.
            slot_dim (int): Dimensionality of each slot vector.
            cond_dim (int): Dimensionality of the conditional input (unused in unconditional version).
        """
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        
        # Encoder: processes each frame with a CNN + positional embedding and MLP.
        self.encoder = Encoder(in_channels=3, feature_dim=slot_dim)
        # Corrector: slot attention module, updates slots based on current frame features.
        self.slot_attention = SlotAttention(num_slots=num_slots, dim=slot_dim, iters=3, hidden_dim=128)
        # Predictor: Transformer-based module to evolve slot states temporally.
        self.predictor = SlotPredictor(slot_dim, hidden_dim=128, num_heads=4)
        # Decoder: spatial broadcast decoder that reconstructs a frame from slots.
        self.decoder = SpatialBroadcastDecoder(
            slot_dim=slot_dim,
            out_channels=3,
            broadcast_resolution=(16, 16),
            final_resolution=(128, 128)
        )
    
    def initialize_slots(self, B, device):
        mu = self.slot_attention.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slot_attention.slots_sigma.exp()
        return mu + sigma * torch.randn(B, self.num_slots, self.slot_dim, device=device)
        
    def forward(self, video, cond_input=None):
        """
        Process a video sequence with a predictor-corrector loop.
        
        Args:
            video: Tensor of shape [B, T, 3, H, W] representing a video clip.
            cond_input: Optional conditional input of shape [B, cond_dim]. (Not used in this unconditional version.)
        
        Returns:
            recon_seq: Tensor of shape [B, T, 3, H, W] with reconstructed frames.
            final_slots: Final slot representations from the last time step.
        """
        B, T, C, H, W = video.shape
        device = video.device
        
        # Initialize slots for time step 1.
        slots = self.initialize_slots(B, device)  # [B, num_slots, slot_dim]
        recon_seq = []
        
        for t in range(T):
            frame = video[:, t]  # Current frame: [B, 3, H, W]
            # Encode the current frame into visual features.
            features = self.encoder(frame)  # [B, N, slot_dim]
            # CORRECTOR: update (correct) slot representations based on current frame,
            # now using the slots from the previous time step.
            corrected_slots = self.slot_attention(features, slots)
            # Decode the corrected slots to obtain the reconstructed frame.
            rgb_slots, alpha_slots = self.decoder(corrected_slots)
            alpha_norm = torch.softmax(alpha_slots, dim=1)
            recon_frame = (rgb_slots * alpha_norm).sum(dim=1)  # [B, 3, H, W]
            recon_seq.append(recon_frame)
            
            # PREDICTOR: if not the last frame, update slots to predict next time step's slots.
            if t < T - 1:
                predicted_slots = self.predictor(corrected_slots)
                slots = predicted_slots  # Use predicted slots as initialization for next frame.
            else:
                slots = corrected_slots
        
        # Stack all reconstructed frames into a video: [B, T, 3, H, W]
        recon_seq = torch.stack(recon_seq, dim=1)
        return recon_seq, slots
