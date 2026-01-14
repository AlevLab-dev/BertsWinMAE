"""
BertsWinMAE: A 3D Masked Autoencoder with BERT-style Swin Transformer

This module implements the BertsWinMAE model, a hybrid architecture combining
a per-patch 3D CNN stem, a BERT-style 3D Swin Transformer encoder, 
and a lightweight 3D CNN decoder.

The architecture is designed for 3D Masked Autoencoding (MAE) pre-training.


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, List, Optional
from functools import partial

# Import Swin components from torchvision
from torchvision.models.video.swin_transformer import SwinTransformerBlock, ShiftedWindowAttention3d
from torchvision.models.swin_transformer import PatchMerging


# ===================================================================
# 1. Internal Helper Functions
# ===================================================================

def _get_3d_sincos_pos_embed(embed_dim: int, grid_size: Tuple[int, int, int]) -> torch.Tensor:
    """
    Generates 3D sinusoidal positional embeddings (SinCos).
    
    Args:
        embed_dim (int): Total embedding dimension (e.g., 768). Must be divisible by 6.
        grid_size (Tuple[int, int, int]): Spatial dimensions of the patch grid (D, H, W).

    Returns:
        torch.Tensor: A tensor of shape (1, N, C), where N is (D*H*W).
    """
    if embed_dim % 6 != 0:
        raise ValueError(
            f"Embedding dim {embed_dim} must be divisible by 6 for 3D "
            "sincos embeddings (Z, Y, X axes, sin/cos pairs)."
        )

    grid_d, grid_h, grid_w = grid_size
    num_patches = grid_d * grid_h * grid_w
    
    dim_per_axis = embed_dim // 3
    
    if dim_per_axis % 2 != 0:
         raise ValueError(f"Per-axis embedding dim ({dim_per_axis}) not even.")

    pos_dim = dim_per_axis // 2 
    omega = torch.exp(torch.arange(pos_dim, dtype=torch.float32) * -(math.log(10000.0) / pos_dim))

    pos_d = torch.arange(grid_d, dtype=torch.float32).unsqueeze(1)
    pos_h = torch.arange(grid_h, dtype=torch.float32).unsqueeze(1)
    pos_w = torch.arange(grid_w, dtype=torch.float32).unsqueeze(1)

    embed_d = torch.cat((torch.sin(pos_d * omega), torch.cos(pos_d * omega)), dim=1)
    embed_h = torch.cat((torch.sin(pos_h * omega), torch.cos(pos_h * omega)), dim=1)
    embed_w = torch.cat((torch.sin(pos_w * omega), torch.cos(pos_w * omega)), dim=1)

    embed_d = embed_d.unsqueeze(1).unsqueeze(1).repeat(1, grid_h, grid_w, 1)
    embed_h = embed_h.unsqueeze(0).unsqueeze(2).repeat(grid_d, 1, grid_w, 1)
    embed_w = embed_w.unsqueeze(0).unsqueeze(0).repeat(grid_d, grid_h, 1, 1)

    pos_embed = torch.cat((embed_d, embed_h, embed_w), dim=-1)
    pos_embed = pos_embed.view(1, num_patches, embed_dim)
    
    return pos_embed

def patchify_image(image_cube: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Divides a 3D image cube into a sequence of flattened patches.

    Args:
        image_cube (torch.Tensor): Input 3D image batch (B, C, D, H, W).
        patch_size (int): The size (P) of the patch.

    Returns:
        torch.Tensor: Sequence of patches (B, N, C*P*P*P).
    """
    p = patch_size
    B, C, D, H, W = image_cube.shape
    if not (D % p == 0 and H % p == 0 and W % p == 0):
        raise ValueError(f"Image dimensions ({D},{H},{W}) not divisible by patch size {p}.")

    n_d, n_h, n_w = D // p, H // p, W // p
    x = image_cube.view(B, C, n_d, p, n_h, p, n_w, p)
    x = x.permute(0, 1, 2, 4, 6, 3, 5, 7)
    patches = x.contiguous().view(B, n_d * n_h * n_w, C * p * p * p)
    
    if C == 1:
        patches = patches.view(B, n_d * n_h * n_w, p * p * p)
        
    return patches

# ===================================================================
# 2. Main Model Class
# ===================================================================

class BertsWinMAE(nn.Module):
    """
    The 3D BertsWin Masked Autoencoder (MAE) model.

    This architecture combines a 3D CNN stem that processes each patch
    individually, a BERT-style 3D Swin Transformer encoder that processes
    the full grid of visible and mask tokens, and a lightweight 3D CNN
    decoder for reconstruction.

    Args:
        img_size (Tuple[int, int, int]): Input image size (D, H, W).
        patch_size (int): Size of the patches (P). Must be a power of 2 (e.g., 8, 16, 32).
        in_chans (int): Number of input channels (default: 1).
        encoder_embed_dim (int): Embedding dimension of the encoder (default: 768).
        encoder_depths (List[int]): List of block counts for each Swin stage (default: [12]).
        encoder_num_heads (List[int]): List of attention heads for each Swin stage (default: [12]).
        swin_window_size (Tuple[int, int, int]): Swin attention window size (default: (7,7,7)).
        decoder_embed_dim (int): Embedding dimension of the decoder (default: 512).
        stem_base_dim (int): Base channel dim for the CNN stem (default: 48).
        mask_ratio (float): Ratio of patches to mask (default: 0.75).
    """
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (224, 224, 224),
        patch_size: int = 16,
        in_chans: int = 1,
        encoder_embed_dim: int = 768,
        encoder_depths: List[int] = [12],
        encoder_num_heads: List[int] = [12],
        swin_window_size: Tuple[int, int, int] = (7, 7, 7),
        decoder_embed_dim: int = 512,
        stem_base_dim: int = 48,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        
        # --- 0. Store parameters and validate ---
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        self.encoder_embed_dim = encoder_embed_dim

        # Validation Check 1: Patch size must be a power of 2
        if not (patch_size > 0 and (patch_size & (patch_size - 1) == 0)):
            raise ValueError(f"patch_size ({patch_size}) must be a power of 2.")
            
        # Validation Check 2: Image dimensions must be divisible by patch size
        if any(i % patch_size != 0 for i in img_size):
            raise ValueError(f"img_size {img_size} dimensions must be divisible by patch_size ({patch_size}).")

        self.grid_size = (
            img_size[0] // patch_size,
            img_size[1] // patch_size,
            img_size[2] // patch_size
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        # Validation Check 3: Swin window size compatibility
        if any(g < w for g, w in zip(self.grid_size, swin_window_size)):
            raise ValueError(
                f"Swin window_size {swin_window_size} cannot be larger than the "
                f"patch grid_size {self.grid_size}."
            )
        
        # Validation Check 4: Enforce non-hierarchical "BERT-style" design
        if len(encoder_depths) > 1 or len(encoder_num_heads) > 1:
            raise ValueError(
                f"BertsWinMAE uses a non-hierarchical design. "
                f"'encoder_depths' and 'encoder_num_heads' must be single-element lists "
                f"(e.g., depths=[12], heads=[12])."
                f"Received depths={encoder_depths}, heads={encoder_num_heads}."
            )

        # --- 1. Per-Patch CNN Stem (Patch Embedding) ---
        self.patch_embed = self._build_patch_embed(
            in_chans=in_chans, 
            out_dim=encoder_embed_dim,
            patch_size=patch_size,
            stem_base_dim=stem_base_dim
        )

        # --- 2. Positional Embedding and Mask Token ---
        pos_embed = _get_3d_sincos_pos_embed(encoder_embed_dim, self.grid_size)
        self.register_buffer('pos_embed', pos_embed)
        self.mask_token_enc = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))

        # --- 3. Swin Transformer Encoder Stages ---
        self.features = self._build_swin_stages(
            embed_dim=encoder_embed_dim,
            depths=encoder_depths,
            num_heads=encoder_num_heads,
            window_size=swin_window_size,
            grid_size=self.grid_size
        )

        # --- 4. Lightweight CNN Decoder ---
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.decoder_head = self._build_lightweight_cnn_head(
            in_dim=decoder_embed_dim,
            out_chans=in_chans,
            patch_size=self.patch_size
        )
        
        # --- 5. Weight Initialization ---
        self.initialize_weights()

    def _build_patch_embed(self, in_chans: int, out_dim: int, patch_size: int, stem_base_dim: int) -> nn.Module:
        """
        Builds the dynamic 3D CNN stem for a SINGLE patch.
        Compresses (in_chans, P, P, P) -> (out_dim, 1, 1, 1).
        
        This dynamically creates log2(P) stride-2 convolution stages.
        For patch_size=16 and stem_base_dim=48, this creates 4 stages
        with channel counts [48, 96, 192, 384], ensuring backward
        compatibility with trained models.
        """
        layers = []
        in_c = in_chans
        current_dim = stem_base_dim
        
        # Calculate number of stages needed to go from P -> 1
        # e.g., P=16 -> log2(16) = 4 stages
        num_stages = int(math.log2(patch_size))

        for i in range(num_stages):
            layers.append(
                nn.Conv3d(in_c, current_dim, kernel_size=3, stride=2, padding=1, bias=False)
            )
            layers.append(nn.GroupNorm(4, current_dim)) 
            layers.append(nn.GELU())
            
            in_c = current_dim
            current_dim = current_dim * 2 # Double channels for next stage

        # Final 1x1 Conv to project to the encoder_embed_dim
        # `in_c` now holds the channel count of the last stage (e.g., 384)
        layers.append(
            nn.Conv3d(in_c, out_dim, kernel_size=1, stride=1)
        )
        return nn.Sequential(*layers)

    def _build_swin_stages(self, embed_dim: int, depths: List[int], 
                           num_heads: List[int], window_size: Tuple[int, int, int],
                           grid_size: Tuple[int, int, int]) -> nn.Module:
        """
        Builds the stack of Swin Transformer blocks.
        """
        layers = []
        in_dim = embed_dim
        current_grid_size = np.array(grid_size) 

        norm_layer = partial(nn.LayerNorm, eps=1e-5)
        block = partial(SwinTransformerBlock, attn_layer=ShiftedWindowAttention3d)
        
        for i_stage in range(len(depths)):
            depth = depths[i_stage]
            n_heads = num_heads[i_stage]
            
            stage_blocks = []
            for i_layer in range(depth):
                shift_size = [0 if i_layer % 2 == 0 else w // 2 for w in window_size]
                
                stage_blocks.append(
                    block(
                        dim=in_dim,
                        num_heads=n_heads,
                        window_size=list(window_size),
                        shift_size=shift_size,
                        mlp_ratio=4.0,
                        dropout=0.0,
                        attention_dropout=0.0,
                        stochastic_depth_prob=0.0,
                        norm_layer=norm_layer,
                    )
                )
            
            layers.append(nn.Sequential(*stage_blocks))
            
            if i_stage < (len(depths) - 1):
                if any(s % 2 != 0 for s in current_grid_size):
                    raise ValueError(f"Grid size {current_grid_size} not divisible by 2 for PatchMerging.")
                    
                layers.append(PatchMerging(in_dim, norm_layer))
                in_dim = in_dim * 2
                current_grid_size = current_grid_size // 2
        
        return nn.Sequential(*layers)

    def _build_lightweight_cnn_head(self, in_dim: int, out_chans: int, patch_size: int) -> nn.Module:
        """
        Builds the lightweight 3D CNN decoder head for upsampling.
        (B, in_dim, D_grid, H_grid, W_grid) -> (B, out_chans, D, H, W)

        This function implements a dynamic 3-stage upsampler that gracefully
        handles any power-of-2 patch size (e.g., 8, 16, 32) while
        remaining 100% backward-compatible with weights trained
        using the original P=16 (s=4,2,2) architecture.
        """
        
        # 1. Calculate the log2 of strides for each of the 3 stages
        total_log_p = int(math.log2(patch_size))
        
        if total_log_p > 6:
            raise ValueError(
                f"Patch size {patch_size} (2^{total_log_p}) is too large "
                f"for the lightweight 3-stage decoder (max 2^6=64)."
            )

        # This logic implements the (2,2,2) -> (4,2,2) -> (4,4,2) -> (4,4,4) pattern
        # Base strides are 1 (log=0)
        l1, l2, l3 = 0, 0, 0
        
        # 1. Base fill (distribute 1s, total_log_p times)
        l1 = min(1, total_log_p); rem = total_log_p - l1
        l2 = min(1, rem); rem = rem - l2
        l3 = min(1, rem); rem = rem - l3
        
        # 2. Top-up fill (distribute remaining logP, topping up to 2)
        add_l1 = min(1, rem); l1 += add_l1; rem -= add_l1
        add_l2 = min(1, rem); l2 += add_l2; rem -= add_l2
        add_l3 = min(1, rem); l3 += add_l3; rem -= add_l3
        
        if rem > 0:
            raise RuntimeError("Internal logic error in decoder stride calculation.")

        # Convert log-strides back to strides (e.g., l=2 -> s=4)
        s1, s2, s3 = 2**l1, 2**l2, 2**l3
        
        # 2. Define the hardcoded, lightweight channels
        # These are constant to ensure backward compatibility and simplicity
        C1, C2 = 64, 16 
        
        # 3. Build the 3 stages
        layers = []
        current_in_dim = in_dim
        
        # --- Stage 1 ---
        if s1 > 1:
            layers.append(nn.ConvTranspose3d(current_in_dim, C1, kernel_size=s1, stride=s1))
        else: # s1=1 (or 0), just do a 1x1 projection
            layers.append(nn.Conv3d(current_in_dim, C1, kernel_size=1, stride=1))
        layers.append(nn.GroupNorm(8, C1)) # 8 groups for 64 channels
        layers.append(nn.GELU())
        current_in_dim = C1

        # --- Stage 2 ---
        if s2 > 0: # This stage is skipped if P < 4
            if s2 > 1:
                layers.append(nn.ConvTranspose3d(current_in_dim, C2, kernel_size=s2, stride=s2))
            else: # s2=1, project
                layers.append(nn.Conv3d(current_in_dim, C2, kernel_size=1, stride=1))
            layers.append(nn.GroupNorm(4, C2)) # 4 groups for 16 channels
            layers.append(nn.GELU())
            current_in_dim = C2
        
        # --- Stage 3 (Final) ---
        if s3 > 0: # This stage is skipped if P < 8
            if s3 > 1:
                layers.append(nn.ConvTranspose3d(current_in_dim, out_chans, kernel_size=s3, stride=s3))
            elif s3 == 1: # s3=1, final project
                layers.append(nn.Conv3d(current_in_dim, out_chans, kernel_size=1, stride=1))
        
        return nn.Sequential(*layers)

    def initialize_weights(self):
        """Initializes model weights."""
        torch.nn.init.normal_(self.mask_token_enc, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
             torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
             if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def random_masking(self, x: torch.Tensor, 
                       valid_for_masking_mask: Optional[torch.Tensor] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs MAE-style random masking.
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        if valid_for_masking_mask is not None:
            noise[~valid_for_masking_mask] = 2.0
            
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x: torch.Tensor, 
                        valid_for_masking_mask: Optional[torch.Tensor] = None
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the BertsWin-style encoder.
        """
        
        # 1. Raw Patchify
        x_patches_raw = patchify_image(x, self.patch_size) 
        
        # 2. Random Masking
        x_visible_raw, mask, ids_restore = self.random_masking(x_patches_raw, valid_for_masking_mask)

        # 3. Embed visible patches
        B, N_vis, P_dim = x_visible_raw.shape
        C = self.encoder_embed_dim
        N = self.num_patches
        
        x_visible_spatial = x_visible_raw.view(B * N_vis, self.in_chans, self.patch_size, self.patch_size, self.patch_size)
        x_visible_embedded = self.patch_embed(x_visible_spatial)
        x_visible = x_visible_embedded.view(B, N_vis, C)

        # 4. Assemble full grid (BERT-style)
        ids_keep = ids_restore.argsort(dim=1)[:, :N_vis]
        x_full = self.mask_token_enc.repeat(B, N, 1).to(dtype=x_visible.dtype)
        x_full.scatter_(dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, C), src=x_visible)
        
        # 5. Add Positional Embeddings
        x = x_full + self.pos_embed
        
        # 6. Run Swin Encoder
        D_grid, H_grid, W_grid = self.grid_size
        x_spatial_in = x.view(B, D_grid, H_grid, W_grid, C) 
        encoded_features_bthwc = self.features(x_spatial_in) 
        encoded_spatial_out = encoded_features_bthwc.permute(0, 4, 1, 2, 3).contiguous()
 
        return encoded_spatial_out, mask
        
    def forward_decoder(self, x_encoded_spatial: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the lightweight CNN decoder.
        """
        x_tokens = x_encoded_spatial.flatten(2).transpose(1, 2) 
        decoded_tokens = self.decoder_embed(x_tokens) 
        B = decoded_tokens.shape[0]

        D_grid, H_grid, W_grid = self.grid_size
        C_dec = self.decoder_embed.out_features

        x_spatial = decoded_tokens.view(B, D_grid, H_grid, W_grid, C_dec)
        x_spatial = x_spatial.permute(0, 4, 1, 2, 3).contiguous()

        reconstructed_cube = self.decoder_head(x_spatial) 
        predictions_patched = patchify_image(reconstructed_cube, self.patch_size)

        return predictions_patched, reconstructed_cube

    def forward(self, x: torch.Tensor, 
                valid_for_masking_mask: Optional[torch.Tensor] = None
                ) -> dict:
        """
        Forward pass for MAE Pre-training.
        
        Args:
            x (torch.Tensor): Input images, shape (B, C, D, H, W).
            valid_for_masking_mask (Optional[torch.Tensor]): Boolean mask 
                (B, N_patches) indicating patches valid for masking.

        Returns:
            dict: A dictionary containing:
                - 'reconstructed_patches': (B, N, P*P*P*C)
                - 'reconstructed_cube': (B, C, D, H, W)
                - 'mae_mask_1d': (B, N), binary mask (1 = masked)
        """
        encoded_features, mask = self.forward_encoder(x, valid_for_masking_mask)
        pred_patches, recon_cube = self.forward_decoder(encoded_features)
        
        return {
            'reconstructed_patches': pred_patches,
            'reconstructed_cube': recon_cube,
            'mae_mask_1d': mask
        }

    def extract_features(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for Feature Extraction (Inference / Downstream tasks).
        Runs the encoder on the full, unmasked image.

        Args:
            x (torch.Tensor): Input images, shape (B, C, D, H, W)
            valid_mask: Boolean mask (B, N_patches). 
                        True = keep original patch (tissue).
                        False = replace with [MASK] token (background).
                        If None, no masking is applied.

        Returns:
            torch.Tensor: The encoded feature maps, e.g., (B, C_enc, D_grid, H_grid, W_grid).
        """
        # 1. Patchify
        x_patches_raw = patchify_image(x, self.patch_size)
        B, N, P_dim = x_patches_raw.shape
        C_enc = self.encoder_embed_dim
        
        # 2. Embed ALL patches
        x_visible_spatial = x_patches_raw.view(B * N, self.in_chans, self.patch_size, self.patch_size, self.patch_size)
        x_visible_embedded = self.patch_embed(x_visible_spatial)
        x_tokens = x_visible_embedded.view(B, N, C_enc)

        if valid_mask is not None:
            # valid_mask shape: (B, N) -> (B, N, C)
            mask_expanded = valid_mask.unsqueeze(-1).expand(-1, -1, C_enc)
            mask_token = self.mask_token_enc.to(x_tokens.dtype) # (1, 1, C)
            x_tokens = torch.where(mask_expanded, x_tokens, mask_token)

        # 3. Add Positional Embedding
        x = x_tokens + self.pos_embed
        
        # 4. Run Swin Encoder
        x_spatial_in = x.view(B, self.grid_size[0], self.grid_size[1], self.grid_size[2], C_enc)
        encoded_features = self.features(x_spatial_in) # (B, D_grid, H_grid, W_grid, C_enc)
        
        # 5. Return features in (B, C, D, H, W) layout
        return encoded_features.permute(0, 4, 1, 2, 3).contiguous()
    
    def extract_volumetric_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts hierarchical features from the full 3D input volume avoiding patch partitioning.

        This method processes the entire input tensor directly through the convolutional 
        stem and Swin Transformer encoder. Unlike the patch-based training routine, 
        this approach preserves the global spatial context during normalization 
        and significantly reduces kernel launch overhead by operating on a single 
        contiguous volume rather than thousands of individual patches.

        Args:
            x (torch.Tensor): Input 3D image tensor with shape (B, C_in, D, H, W).
                              Dimensions (D, H, W) must be divisible by the patch size.

        Returns:
            torch.Tensor: Encoded spatiotemporal features with shape (B, C_enc, D', H', W'),
                          where (D', H', W') = (D, H, W) / patch_size.
        """
        # 1. Global Convolutional Stem
        # Projects high-res input directly to feature space: 
        # (B, C_in, D, H, W) -> (B, C_enc, D/p, H/p, W/p)
        x_stem = self.patch_embed(x)

        # 2. Tensor Permutation for Transformer Compatibility
        # Swin Transformer expects channel-last format: (B, D', H', W', C_enc)
        x_spatial = x_stem.permute(0, 2, 3, 4, 1).contiguous()
        
        B, D_grid, H_grid, W_grid, C_enc = x_spatial.shape

        # 3. Add Positional Embeddings
        # Flatten spatial dims to (B, N, C) for broadcasting with pos_embed (1, N, C)
        x_tokens = x_spatial.view(B, -1, C_enc)
        x_tokens = x_tokens + self.pos_embed

        # 4. Swin Transformer Encoder
        # Reshape back to grid for window-based attention mechanism
        x_swin_input = x_tokens.view(B, D_grid, H_grid, W_grid, C_enc)
        encoded_features_swin = self.features(x_swin_input)

        # 5. Feature Reconstruction
        # Restore standard PyTorch channel-first layout: (B, C_enc, D', H', W')
        return encoded_features_swin.permute(0, 4, 1, 2, 3).contiguous()

# ===================================================================
# 3. Factory Functions (TIMM-style)
# ===================================================================

def bertswin_mae_base_patch16_224(**kwargs):
    """
    Creates a BertsWinMAE-Base model with 224^3 input and 16^3 patch size.
    (Encoder: 768 dim, 12 depth. Decoder: 512 dim. StemBase: 48)
    """
    model = BertsWinMAE(
        img_size=(224, 224, 224),
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depths=[12],
        encoder_num_heads=[12],
        swin_window_size=(7, 7, 7),
        decoder_embed_dim=512,
        stem_base_dim=48, # This ensures compatibility
        **kwargs  # Allows overriding any parameter
    )
    return model