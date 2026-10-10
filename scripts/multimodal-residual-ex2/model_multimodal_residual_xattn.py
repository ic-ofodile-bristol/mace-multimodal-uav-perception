#!/usr/bin/env python3
"""
Frozen MViTv2-S + frozen Ex3-binary aircraft Transformer + gated cross-attention.

Vision:
    16 frames -> fine-tuned MViTv2-S backbone -> z_v [B,768]
Aircraft:
    50x8 -> 25 patches -> binary-pretrained Transformer -> H_a [B,25,64]
Fusion:
    z_v is a single query; H_a supplies ordered keys/values.
    Multi-head cross-attention dimension = 256.
    Attention result is projected back to vision dimension and gated into z_v.
Output:
    four semantic interaction classes.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models.video import mvit_v2_s

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
KINETICS_MEAN = (0.45, 0.45, 0.45)
KINETICS_STD = (0.225, 0.225, 0.225)

class AircraftTokenEncoder(nn.Module):
    """Exact token-producing portion of the Ex3 binary TimeMILPatchTransformer."""
    def __init__(
        self, num_features=8, window_steps=50, patch_len=2, patch_stride=2,
        d_model=64, num_heads=4, num_layers=2, ff_dim=128, dropout=0.40,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.num_patches = 1 + (window_steps - patch_len) // patch_stride
        self.patch_projection = nn.Sequential(
            nn.Linear(num_features * patch_len, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )

    def make_patches(self, x):
        patches = x.unfold(1, self.patch_len, self.patch_stride)
        return patches.permute(0,1,3,2).contiguous().view(x.shape[0], -1, x.shape[2]*self.patch_len)

    def forward(self, x):
        tokens = self.patch_projection(self.make_patches(x))
        tokens = tokens + self.position_embedding[:, :tokens.shape[1]]
        return self.transformer(tokens)

class MViTFeatureEncoder(nn.Module):
    """MViTv2-S feature encoder matching the user's Ex5 wrapper."""
    def __init__(self):
        super().__init__()
        backbone = mvit_v2_s(weights=None)
        final_linear = None
        if isinstance(backbone.head, nn.Sequential):
            for layer in reversed(backbone.head):
                if isinstance(layer, nn.Linear):
                    final_linear = layer
                    break
        elif isinstance(backbone.head, nn.Linear):
            final_linear = backbone.head
        if final_linear is None:
            raise RuntimeError("Could not identify MViTv2-S final linear layer.")
        self.embedding_dimension = int(final_linear.in_features)
        backbone.head = nn.Identity()
        self.backbone = backbone

        self.register_buffer("imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1,1,3,1,1))
        self.register_buffer("imagenet_std", torch.tensor(IMAGENET_STD).view(1,1,3,1,1))
        self.register_buffer("kinetics_mean", torch.tensor(KINETICS_MEAN).view(1,1,3,1,1))
        self.register_buffer("kinetics_std", torch.tensor(KINETICS_STD).view(1,1,3,1,1))

    def prepare_video(self, frames):
        # dataset_online_final: [B,T,C,H,W], ImageNet normalised.
        if frames.ndim != 5 or frames.shape[1] != 16 or frames.shape[2] != 3:
            raise ValueError(f"Expected [B,16,3,H,W], got {tuple(frames.shape)}")
        frames = frames.float()
        rgb = (frames * self.imagenet_std + self.imagenet_mean).clamp(0,1)
        kinetics = (rgb - self.kinetics_mean) / self.kinetics_std
        return kinetics.permute(0,2,1,3,4).contiguous()

    def forward(self, frames):
        return self.backbone(self.prepare_video(frames))

def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint

def load_vision_backbone(encoder, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _extract_state_dict(ckpt)
    # Ex5 wrapper stores backbone.* plus classifier.* and normalisation buffers.
    backbone_state = {
        k[len("backbone."):]: v for k,v in state.items()
        if k.startswith("backbone.")
    }
    if not backbone_state:
        # Also tolerate DataParallel wrapper.
        backbone_state = {
            k[len("module.backbone."):]: v for k,v in state.items()
            if k.startswith("module.backbone.")
        }
    if not backbone_state:
        raise RuntimeError("No 'backbone.' keys found in vision checkpoint.")
    missing, unexpected = encoder.backbone.load_state_dict(backbone_state, strict=False)
    # head is Identity, so old classifier-head keys should not be present in backbone_state.
    if unexpected:
        raise RuntimeError(f"Unexpected vision backbone keys: {unexpected}")
    return missing

def load_aircraft_encoder(encoder, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _extract_state_dict(ckpt)
    wanted_prefixes = ("patch_projection.", "position_embedding", "transformer.")
    aircraft_state = {k:v for k,v in state.items() if k.startswith(wanted_prefixes)}
    if not aircraft_state:
        aircraft_state = {
            k[len("module."):]:v for k,v in state.items()
            if k.startswith("module.") and k[len("module."):].startswith(wanted_prefixes)
        }
    missing, unexpected = encoder.load_state_dict(aircraft_state, strict=True)
    return missing, unexpected

class MACEResidualGatedCrossAttention(nn.Module):
    def __init__(
        self, vision_checkpoint, aircraft_checkpoint,
        attention_dim=256, attention_heads=4, dropout=0.30,
        num_classes=4, freeze_encoders=True, residual_scale_init=0.10,
    ):
        super().__init__()
        self.vision_encoder = MViTFeatureEncoder()
        self.aircraft_encoder = AircraftTokenEncoder()

        load_vision_backbone(self.vision_encoder, vision_checkpoint)
        load_aircraft_encoder(self.aircraft_encoder, aircraft_checkpoint)

        vdim = self.vision_encoder.embedding_dimension
        adim = 64
        self.query_projection = nn.Linear(vdim, attention_dim)
        self.key_projection = nn.Linear(adim, attention_dim)
        self.value_projection = nn.Linear(adim, attention_dim)
        self.cross_attention = nn.MultiheadAttention(
            attention_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.attention_to_vision = nn.Linear(attention_dim, vdim)

        # Feature-wise gate: each visual dimension decides how much telemetry
        # evidence to admit.
        self.gate = nn.Sequential(
            nn.Linear(vdim * 2, vdim),
            nn.Sigmoid(),
        )
        # Ex2: telemetry is a small learned residual correction to vision.
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.float32)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(vdim),
            nn.Dropout(dropout),
            nn.Linear(vdim, num_classes),
        )

        self.freeze_encoders = bool(freeze_encoders)
        if self.freeze_encoders:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            for p in self.aircraft_encoder.parameters():
                p.requires_grad = False
            self.vision_encoder.eval()
            self.aircraft_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoders:
            self.vision_encoder.eval()
            self.aircraft_encoder.eval()
        return self

    def forward(self, video, telemetry, return_attention=False):
        if self.freeze_encoders:
            with torch.no_grad():
                z_v = self.vision_encoder(video)       # [B,V]
                h_a = self.aircraft_encoder(telemetry) # [B,25,64]
        else:
            z_v = self.vision_encoder(video)
            h_a = self.aircraft_encoder(telemetry)

        q = self.query_projection(z_v).unsqueeze(1)  # [B,1,D]
        k = self.key_projection(h_a)                 # [B,25,D]
        v = self.value_projection(h_a)               # [B,25,D]

        attended, weights = self.cross_attention(
            q, k, v, need_weights=True, average_attn_weights=False
        )
        attended = self.attention_to_vision(attended.squeeze(1))
        gate = self.gate(torch.cat([z_v, attended], dim=-1))
        residual = gate * attended
        fused = z_v + self.residual_scale * residual
        logits = self.classifier(fused)

        if return_attention:
            return {
                "logits": logits,
                "attention": weights,  # [B,heads,1,25]
                "gate": gate,
                "residual_scale": self.residual_scale,
                "residual_features": residual,
                "vision_features": z_v,
                "aircraft_tokens": h_a,
                "fused_features": fused,
            }
        return logits
