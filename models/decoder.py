# models/decoder.py
import torch
import torch.nn as nn

class SpatialBroadcastDecoder(nn.Module):
    def __init__(self, slot_dim, out_channels=3, broadcast_resolution=(16, 16), final_resolution=(128, 128)):
        super().__init__()
        self.broadcast_resolution = broadcast_resolution
        self.final_resolution = final_resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(slot_dim + 2, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 4, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, slots):
        B, K, D = slots.shape
        H, W = self.broadcast_resolution
        slots = slots.view(B * K, D)
        slot_broadcast = slots.unsqueeze(-1).unsqueeze(-1).expand(B * K, D, H, W)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, H, device=slots.device),
            torch.linspace(0, 1, W, device=slots.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B * K, -1, -1, -1)
        x = torch.cat([slot_broadcast, grid], dim=1)
        out = self.decoder(x)
        rgb = out[:, :-1, :, :]
        alpha = out[:, -1:, :, :]
        final_H, final_W = self.final_resolution
        rgb = rgb.view(B, K, rgb.shape[1], final_H, final_W)
        alpha = alpha.view(B, K, 1, final_H, final_W)
        return rgb, alpha
