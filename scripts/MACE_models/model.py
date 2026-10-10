#!/usr/bin/env python3
"""
===============================================================================
model.py
===============================================================================

Vision-only baseline model.

Module 1
---------
Pretrained Vision Transformer backbone.

Responsibilities
----------------
- Load pretrained ViT-B/16
- Remove classification head
- Produce a 768-dimensional embedding for each frame
===============================================================================
"""

import torch
import torch.nn as nn

from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights,
)


# =============================================================================
# Configuration
# =============================================================================

EMBEDDING_DIM = 768
NUM_CLASSES = 4

# =============================================================================
# Vision Backbone
# =============================================================================

class VisionBackbone(nn.Module):
    """
    Pretrained Vision Transformer backbone.

    Input
    -----
    (N,3,224,224)

    Output
    ------
    (N,768)
    """

    def __init__(self):

        super().__init__()

        weights = ViT_B_16_Weights.IMAGENET1K_V1

        self.backbone = vit_b_16(weights=weights)

        # Remove ImageNet classifier
        self.backbone.heads = nn.Identity()

    # -------------------------------------------------------------------------

    def forward(self, x):

        """
        Parameters
        ----------
        x
            Tensor (N,3,224,224)

        Returns
        -------
        Tensor (N,768)
        """

        embedding = self.backbone(x)

        return embedding


# =============================================================================
# Vision Classifier
# =============================================================================

class VisionClassifier(nn.Module):
    """
    Vision-only baseline classifier.

    Input
    -----
    (B,25,3,224,224)

    Output
    ------
    (B,4)
    """

    def __init__(self):

        super().__init__()

        self.backbone = VisionBackbone()

        self.classifier = nn.Sequential(

            nn.Linear(EMBEDDING_DIM, 512),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(512, NUM_CLASSES)

        )

    # -------------------------------------------------------------------------

    def forward(self, x):

        """
        Parameters
        ----------
        x
            (B,T,C,H,W)

        Returns
        -------
        logits
            (B,4)
        """

        batch_size, sequence_length, channels, height, width = x.shape

        # --------------------------------------------------------------
        # Flatten temporal dimension
        # --------------------------------------------------------------

        x = x.view(
            batch_size * sequence_length,
            channels,
            height,
            width
        )

        # --------------------------------------------------------------
        # Extract frame embeddings
        # --------------------------------------------------------------

        frame_embeddings = self.backbone(x)

        # --------------------------------------------------------------
        # Restore temporal structure
        # --------------------------------------------------------------

        frame_embeddings = frame_embeddings.view(
            batch_size,
            sequence_length,
            EMBEDDING_DIM
        )

        # --------------------------------------------------------------
        # Temporal aggregation
        # --------------------------------------------------------------

        sequence_embedding = frame_embeddings.mean(dim=1)

        # --------------------------------------------------------------
        # Classification
        # --------------------------------------------------------------

        logits = self.classifier(sequence_embedding)

        return logits
    


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":

    model = VisionClassifier()

    # 8 25 3
    dummy = torch.randn(
        1,
        2,
        3,
        224,
        224,
    )

    logits = model(dummy)

    print("\nModel Output")
    print("---------------------")
    print("Input :", dummy.shape)
    print("Output:", logits.shape)