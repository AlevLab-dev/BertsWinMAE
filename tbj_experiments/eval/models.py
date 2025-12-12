import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple, List

try:
    from monai.networks.nets.swin_unetr import SwinTransformer
except ImportError:
    SwinTransformer = None

try:
    from segment_anything.build_sam3D import sam_model_registry3D
except ImportError:
    sam_model_registry3D = None



class LoRALinear(nn.Linear):
    """
    Replaces nn.Linear with a LoRA-enabled layer.
    
    This class inherits from nn.Linear, so it can be seamlessly
    swapped while preserving attributes like .weight and .bias,
    which is crucial for torchvision's internal SwinTransformer code.
    
    The original weights (self.weight, self.bias) are frozen.
    Two new trainable parameters (lora_A, lora_B) are added.
    
    Forward pass computes: y = F.linear(x, W_frozen, b_frozen) + (x @ A @ B) * scale
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int = 8,
        alpha: int = 16,
        device=None,
        dtype=None,
    ):
        # Initialize the parent nn.Linear layer
        # This creates self.weight and self.bias on the correct device/dtype
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)

        self.lora_rank = rank
        self.lora_alpha = alpha
        
        # Create new LoRA parameters on the same device/dtype
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.lora_A = nn.Parameter(torch.empty(in_features, rank, **factory_kwargs))
        self.lora_B = nn.Parameter(torch.empty(rank, out_features, **factory_kwargs))
        
        # Initialize LoRA parameters
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the LoRA forward pass.
        """
        # 1. Frozen pass using the original weights (self.weight, self.bias)
        #    This is what the parent nn.Linear.forward() does.
        y_frozen = F.linear(x, self.weight, self.bias)
        
        # 2. Trainable LoRA pass
        y_lora = (x @ self.lora_A @ self.lora_B) * (self.lora_alpha / self.lora_rank)
        
        return y_frozen + y_lora

    def __repr__(self):
        s = super().__repr__()
        # Add LoRA info to the default nn.Linear repr
        s = s.replace(')', f', lora_rank={self.lora_rank}, lora_alpha={self.lora_alpha})')
        return s

def inject_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    target_modules_keywords: List[str] = ['attn', 'mlp']
):
    """
    Recursively traverses the model and replaces specified nn.Linear
    layers with LoRALinear layers, correctly copying weights and device.
    """
    
    layers_to_replace = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
            
        if any(keyword in name for keyword in target_modules_keywords):
            layers_to_replace.append(name)

    replaced_count = 0
    for name in layers_to_replace:
        try:
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = model.get_submodule(parent_name)
            
            # Get the original nn.Linear layer
            original_linear_layer = getattr(parent_module, child_name)
            
            if isinstance(original_linear_layer, LoRALinear):
                continue # Already replaced

            # --- Key Fix: Get device and dtype from the existing layer ---
            in_f = original_linear_layer.in_features
            out_f = original_linear_layer.out_features
            has_bias = original_linear_layer.bias is not None
            
            factory_kwargs = {
                'device': original_linear_layer.weight.device,
                'dtype': original_linear_layer.weight.dtype
            }

            # Create the new LoRALinear layer
            new_lora_layer = LoRALinear(
                in_features=in_f,
                out_features=out_f,
                bias=has_bias,
                rank=rank,
                alpha=alpha,
                **factory_kwargs # Pass device and dtype
            )
            
            # Copy the original weights and bias, then freeze them
            new_lora_layer.weight.data.copy_(original_linear_layer.weight.data)
            new_lora_layer.weight.requires_grad = False
            
            if has_bias:
                new_lora_layer.bias.data.copy_(original_linear_layer.bias.data)
                new_lora_layer.bias.requires_grad = False
            
            # Replace the old layer with the new LoRA-enabled layer
            setattr(parent_module, child_name, new_lora_layer)
            replaced_count += 1
            
        except Exception as e:
            logging.warning(f"Failed to inject LoRA into '{name}': {e}")
            
    if replaced_count == 0:
        logging.warning(f"LoRA: No nn.Linear layers were replaced. "
                        f"Keywords: {target_modules_keywords}. Model: {model.__class__.__name__}")
    else:
        logging.info(f"LoRA: Successfully replaced {replaced_count} nn.Linear layers.")

class GeneralizedMeanPooling3d(nn.Module):
    """
    Trainable pooling layer that interpolates between Mean (p=1) and Max (p->inf) pooling.
    Focuses on regions with high activation intensity (salient features).
    Formula: x = (1/N * sum(|x|^p))^(1/p)
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W]
        x_abs = x.abs().clamp(min=self.eps)
        
        x_pow = x_abs.pow(self.p)
        avg = F.avg_pool3d(x_pow, kernel_size=x.shape[2:]).view(x.size(0), -1)
        return avg.pow(1.0 / self.p)

class TextureHead(nn.Module):
    """
    Head for Inflammation detection.
    Aggregates local texture anomalies using GeM and Standard Deviation pooling.
    Input: [B, C, D, H, W] -> Output: [B, 1] logits
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.accepts_spatial_features = True # Flag for TMJAnalysisModel
        
        reduced_dim = 128
        
        # 1x1 Conv to compress features before pooling
        self.bottleneck = nn.Conv3d(in_features, reduced_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(reduced_dim)
        
        self.gem_pool = GeneralizedMeanPooling3d(p=3.0)
        
        self.fc = nn.Sequential(
            nn.Linear(reduced_dim * 2, 128), 
            nn.LayerNorm(128),
            nn.Dropout(0.3),
            nn.GELU(), 
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Compress features (Linear projection of pixels)
        # Output can be positive or negative, allowing GeM to see "negative energy".
        x = self.bn(self.bottleneck(x))
        
        # 2. GeM Pooling (Intensity/Energy)
        # Uses |x|^p, capturing magnitude of signals
        f_gem = self.gem_pool(x) # [B, 128]
        
        # 3. Std Deviation Pooling (Heterogeneity/Texture)
        # Calculates spatial variance. Variance is independent of sign.
        f_std = torch.std(x, dim=(-3, -2, -1), unbiased=False) # [B, 128]
        
        # 4. Fusion
        f_cat = torch.cat([f_gem, f_std], dim=1)
        return self.fc(f_cat)

class SpatialAttentionHead(nn.Module):
    """
    Head for Bone and Disc pathology.
    Uses 3D Attention to weigh critical anatomical regions before pooling.
    Input: [B, C, D, H, W] -> Output: [B, 1] logits
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.accepts_spatial_features = True
        self.attention_net = nn.Sequential(
            nn.Conv3d(in_features, in_features // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_features // 4),
            nn.Tanh(),
            nn.Conv3d(in_features // 4, 1, kernel_size=1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.Dropout(0.25),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Generate Attention Map [B, 1, D, H, W]
        attn_logits = self.attention_net(x)
        attn_weights = F.softmax(attn_logits.flatten(2), dim=2).view_as(attn_logits)
        
        # 2. Weighted Pooling
        x_weighted = x * attn_weights
        x_pooled = x_weighted.sum(dim=(-3, -2, -1)) # Global Sum over spatial dims
        
        return self.fc(x_pooled)

class VolumetricPyramidHead(nn.Module):
    """
    Head for Joint Space Regression.
    Preserves coarse spatial geometry using fixed-grid pooling.
    Input: [B, C, D, H, W] -> Output: [B, 1] mm
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.accepts_spatial_features = True
        # Reduces 14x14x14 -> 4x4x4 (approx 3.2mm -> 11mm grid cells)
        self.grid_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        
        flat_dim = in_features * 4 * 4 * 4
        
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_grid = self.grid_pool(x) # [B, C, 4, 4, 4]
        x_flat = x_grid.flatten(1) # [B, C*64]
        return self.fc(x_flat)

class ModelFactory:
    """
    A unified factory for constructing backbone encoders.
    
    Key Responsibility:
    Ensures strict adherence to pre-training protocols. This includes:
    1. Matching architecture configurations (BERT-style vs ViT vs Swin).
    2. Handling 'Surgical Weight Transfer' for architecturally incompatible models (e.g., MAE -> Classifier).
    3. Filtering checkpoint keys for partial loading (e.g., extracting only the encoder from a full UNet).
    """
    
    @staticmethod
    def load_backbone(
        arch_type: str, 
        checkpoint_path: Optional[str], 
        config: Any, 
        pretrain_modules: Dict[str, Any] = None
    ) -> Tuple[nn.Module, int]:
        """
        Instantiates the specific encoder architecture and loads weights if provided.
        
        Args:
            arch_type: One of ['bertswin_mae', 'monai_vit', 'monai_swin_unetr_btcv', 'sam_med3d_vit_b'].
            checkpoint_path: Path to .pth file or None (for random initialization).
            config: The hydrated configuration object.
            pretrain_modules: Dictionary containing dynamically imported classes from the pre-training script.
        
        Returns:
            backbone (nn.Module): The image encoder.
            feature_dim (int): The output dimension of the global embedding.
        """
        backbone = None
        feature_dim = 0
        is_random = (checkpoint_path is None)
        
        logging.info(f"Initializing backbone: {arch_type} (Pre-trained: {not is_random})")

        # ======================================================================
        # 1. BertsWinMAE (Custom BERT-style Transformer)
        # ======================================================================
        if arch_type == 'bertswin_mae':
            if pretrain_modules is None:
                raise ValueError("pretrain_modules is required for BertsWinMAE initialization.")
            
            BertsWinMAE = pretrain_modules['BertsWinMAE']
            backbone = BertsWinMAE(
                img_size=config.INPUT_CUBE_SIZE,              # (224, 224, 224)
                patch_size=config.EFFECTIVE_PATCH_SIZE,       # 16
                in_chans=1,
                encoder_embed_dim=config.ENCODER_EMBED_DIM,   # 768
                encoder_depths=config.SWIN_DEPTHS,            # [12]
                encoder_num_heads=config.SWIN_NUM_HEADS,      # [12]
                swin_window_size=config.SWIN_WINDOW_SIZE,     # (7, 7, 7)
                decoder_embed_dim=config.DECODER_EMBED_DIM,   # 512
                stem_base_dim=config.STEM_BASE_DIM,           # 48
                mask_ratio=config.MAE_MASK_RATIO              # 0.75
            )
            feature_dim = config.ENCODER_EMBED_DIM
            
            if not is_random:
                ModelFactory._load_checkpoint(backbone, checkpoint_path, key_search=['model_state_dict', 'model'])

        # ======================================================================
        # 2. MONAI ViT (Standard ViT with MAE Surgical Weight Transfer)
        # ======================================================================
        elif arch_type == 'monai_vit':
            # ACADEMIC NOTE: SURGICAL WEIGHT TRANSFER & POSITIONAL EMBEDDINGS
            # ----------------------------------------------------------------------
            # We address a fundamental architectural incompatibility between the 
            # MONAI `MaskedAutoEncoderViT` (used for pre-training) and the standard `ViT` 
            # (used for classification).
            #
            # 1. Embedding Mismatch: MAE uses non-learnable 'sincos' embeddings by default, 
            #    whereas standard ViT defaults to random 'learnable' embeddings. 
            #    Direct weight loading results in random position information (AUC ~0.5).
            #    We force `pos_embed_type="sincos"` to match the pre-trained latent space.
            #
            # 2. LayerNorm Mismatch: MAE encapsulates the final norm within a Sequential block,
            #    while ViT expects a standalone `norm` attribute. We perform manual parameter mapping.
            #
            # 3. CLS Token: The [CLS] token is NOT optimized during MAE reconstruction 
            #    pre-training. We strictly ignore it during downstream pooling.
            # ----------------------------------------------------------------------
            if pretrain_modules is None:
                raise ValueError("pretrain_modules is required for MONAI ViT initialization.")
            
            MonaiMAEWrapper = pretrain_modules['MonaiMAEWrapper']
            MonaiViT = pretrain_modules['MonaiViT']
            
            # A. Load Source Weights (The MAE Wrapper)
            # We must instantiate the full MAE to correctly load the 'sincos' weights.
            mae_model = MonaiMAEWrapper(config)
            if not is_random:
                ModelFactory._load_checkpoint(mae_model, checkpoint_path, key_search=['model_state_dict'])
            
            # B. Initialize Target Model (The Classifier ViT)
            # CRITICAL: If loading from MAE, we must force 'sincos' pos_embed_type.
            # Standard ViT defaults to 'learnable', which would remain random if not matched.
            pos_embed_type = "sincos" if not is_random else "learnable"
            
            backbone = MonaiViT(
                in_channels=1,
                img_size=config.INPUT_CUBE_SIZE,
                patch_size=config.MONAI_VIT_PATCH_SIZE,
                hidden_size=config.MONAI_VIT_EMBED_DIM,
                mlp_dim=config.MONAI_VIT_MLP_DIM,
                num_layers=config.MONAI_VIT_DEPTH,
                num_heads=config.MONAI_VIT_NUM_HEADS,
                pos_embed_type=pos_embed_type, 
                classification=True,
                spatial_dims=3
            )
            
            # C. Surgical Weight Transfer
            if not is_random:
                logging.info("Performing surgical weight transfer: MAE Encoder -> ViT Classifier")
                mae_encoder = mae_model.mae
                
                # 1. Patch Embeddings
                backbone.patch_embedding.load_state_dict(mae_encoder.patch_embedding.state_dict())
                # 2. Transformer Blocks (N-1 blocks in MAE are standard transformer blocks)
                backbone.blocks.load_state_dict(mae_encoder.blocks[:-1].state_dict())
                # 3. Final Norm (The last block in MAE encoder is actually the LayerNorm)
                backbone.norm.load_state_dict(mae_encoder.blocks[-1].state_dict())
                
                # Note: The CLS token is NOT transferred. In MAE, the CLS token is not 
                # used for reconstruction loss, thus it learns no semantic information.
                # We leave the ViT CLS token randomly initialized (and ignore it during pooling).
            
            feature_dim = config.MONAI_VIT_EMBED_DIM

        # ======================================================================
        # 3. Swin UNETR (BTCV Pre-trained)
        # ======================================================================
        elif arch_type == 'monai_swin_unetr_btcv':
            if SwinTransformer is None:
                raise ImportError("monai.networks.nets.swin_unetr.SwinTransformer is required.")

            # BTCV Pre-training Configuration (Fixed architecture parameters)
            # Note: Input size handling (96 vs 224) happens in TMJAnalysisModel.
            backbone = SwinTransformer(
                in_chans=1,
                embed_dim=48,
                window_size=(7, 7, 7),
                patch_size=(2, 2, 2),
                depths=(2, 2, 2, 2),
                num_heads=(3, 6, 12, 24),
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                norm_layer=nn.LayerNorm,
                use_checkpoint=False,
                spatial_dims=3,
                downsample="merging"
            )
            
            # Feature dim calculation: embed_dim * 2^(num_layers) = 48 * 16 = 768
            feature_dim = 768 
            
            if not is_random:
                # Filter weights to extract only the 'swinViT' submodule from the full UNETR
                ModelFactory._load_checkpoint(backbone, checkpoint_path, prefix_filter='swinViT')

        # ======================================================================
        # 4. SAM Med3D (vit_b_ori)
        # ======================================================================
        elif arch_type == 'sam_med3d_vit_b':
            if sam_model_registry3D is None:
                raise ImportError("segment_anything.build_sam3D is required.")
            
            # Instantiate full SAM model
            full_sam_model = sam_model_registry3D['vit_b_ori'](checkpoint=None)
            # Extract only the image encoder
            backbone = full_sam_model.image_encoder
            
            # SAM ViT-B output dimension
            feature_dim = 384 
            
            if not is_random:
                # Filter weights to extract only 'image_encoder'
                ModelFactory._load_checkpoint(
                    backbone, 
                    checkpoint_path, 
                    key_search=['model_state_dict'],
                    prefix_filter='image_encoder'
                )

        else:
            raise ValueError(f"Unknown architecture type: {arch_type}")
            
        return backbone, feature_dim

    @staticmethod
    def _load_checkpoint(model: nn.Module, path: str, key_search: list = None, prefix_filter: str = None):
        """
        Robust checkpoint loader handling DDP prefixes and partial dictionary matching.
        """
        logging.info(f"Loading weights from {path} ...")
        try:
            state_full = torch.load(path, map_location='cpu')
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        # 1. Unwrap state dict from potential wrappers ('model', 'state_dict')
        state_dict = state_full
        if key_search:
            for k in key_search:
                if isinstance(state_full, dict) and k in state_full:
                    state_dict = state_full[k]
                    break
        
        # 2. Process keys (Remove DDP prefixes, Apply filter)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # Remove DDP artifacts
            clean_k = k.replace('module.', '').replace('_orig_mod.', '')
            
            if prefix_filter:
                if clean_k.startswith(prefix_filter):
                    # Strip the filter prefix (e.g. 'swinViT.block' -> 'block')
                    # We add +1 for the dot separator
                    clean_k = clean_k[len(prefix_filter)+1:]
                else:
                    continue # Skip keys not belonging to the submodule
            
            new_state_dict[clean_k] = v
            
        # 3. Load weights into the model
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        
        if len(missing) > 0:
            logging.warning(f"Missing keys during load (Top 5): {missing[:5]}")
        if len(unexpected) > 0:
            logging.debug(f"Unexpected keys during load (Top 5): {unexpected[:5]}")
        
        logging.info("Weights loaded successfully.")


class TMJAnalysisModel(nn.Module):
    """
    Unified Evaluator for Bilateral TMJ Analysis.
    
    Modes:
    1. Bilateral Aggregation (default): Fuses L and R features (max/mean) for patient-level tasks.
    2. Independent Processing: If x_pair is None, processes x through the head directly. 
       This allows treating each joint as an independent sample (batch * 2) for joint-level analysis.
    """
    def __init__(
        self, 
        backbone: nn.Module, 
        head: nn.Module, 
        feature_dim: int, 
        arch_type: str, 
        inference_mode: str = 'full_image',
        patchify_fn: Optional[callable] = None,
        patch_size: Optional[int] = 16
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.arch_type = arch_type
        self.inference_mode = inference_mode
        self.feature_dim = feature_dim
        self.patchify_fn = patchify_fn
        self.patch_size = patch_size
        self.pool_3d = nn.AdaptiveAvgPool3d(1)
    
    def _reshape_tokens_to_volume(self, x_tokens: torch.Tensor) -> torch.Tensor:
        """
        Helper: Reshapes flat transformer tokens (B, N, C) back to 3D volume (B, C, S, S, S).
        Assumes a cubic grid.
        """
        # 
        if x_tokens.ndim == 4 and x_tokens.shape[1] == 1:
             # Sometimes outputs come as (B, 1, N, C) from libraries
             x_tokens = x_tokens.squeeze(1)
             
        B, N, C = x_tokens.shape
        S = int(round(N ** (1/3)))
        
        if S**3 != N:
            raise ValueError(f"Token count {N} cannot be reshaped into a perfect cube (S={S}). Check patch size or input shape.")
            
        # Permute: (B, N, C) -> (B, C, N)
        # View: (B, C, S, S, S)
        return x_tokens.permute(0, 2, 1).contiguous().view(B, C, S, S, S)

    def get_embedding(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for a single branch (Left or Right).
        Contains specific logic to adapt input size and pool features based on the architecture.
        """
        spatial_features = None
        # A. BertsWinMAE (Custom BERT-style)
        if self.arch_type == 'bertswin_mae':
            # Handle Background Masking Inference Mode
            valid_mask = None
            if self.inference_mode == 'background_mask' and mask is not None:
                if mask.dim() == 2:
                    # Mask is already [B, N_patches], use it directly
                    valid_mask = mask.bool()
                else:
                    if self.patchify_fn is None:
                        raise ValueError("Inference mode 'background_mask' requires 'patchify_fn'.")
                    
                    # Ensure mask has channel dim [B, 1, D, H, W] for patchify
                    if mask.dim() == 4:
                        mask = mask.unsqueeze(1)
                    
                    # Dimensions: (B, 1, D, H, W) -> (B, N_patches, P*P*P)
                    mask_patches = self.patchify_fn(mask.float(), self.patch_size)
                    # Calculate ratio of valid data per patch
                    data_ratio = mask_patches.mean(dim=-1)
                    valid_mask = (data_ratio > 0.1)

            # The backbone.extract_features returns (B, C, D, H, W).
            spatial_features = self.backbone.extract_features(x, valid_mask=valid_mask)
            
        # B. MONAI ViT
        elif self.arch_type == 'monai_vit':
            # Logic: Add CLS -> Encoder -> Ignore CLS -> GAP on patch tokens
            # This matches the pre-training reconstruction logic where the CLS token is unused.
            x_emb = self.backbone.patch_embedding(x)
            cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
            x_tokens = torch.cat((cls_token, x_emb), dim=1)
            
            # Pass through Transformer Blocks
            for blk in self.backbone.blocks:
                x_tokens = blk(x_tokens)
            x_tokens = self.backbone.norm(x_tokens)

            # 2. Remove CLS token -> [B, N_patches, C]
            patch_tokens = x_tokens[:, 1:, :]
            spatial_features = self._reshape_tokens_to_volume(patch_tokens)
            
        # C. Swin UNETR (BTCV)
        elif self.arch_type == 'monai_swin_unetr_btcv':
            # BTCV pre-trained weights expect input size (96, 96, 96).
            # If input is larger (e.g., 224), we interpolate to prevent performance degradation.
            if x.shape[2] > 96:
                x = F.interpolate(x, size=(96, 96, 96), mode='trilinear', align_corners=False)
            
            # Forward returns a list of feature maps. We take the last one (bottleneck).
            features_list = self.backbone(x)
            spatial_features = features_list[-1]
            
        # D. SAM Med3D
        elif self.arch_type == 'sam_med3d_vit_b':
            # SAM Med3D expects input size (128, 128, 128).
            if x.shape[2] != 128:
                x = F.interpolate(x, size=(128, 128, 128), mode='trilinear', align_corners=False)
            
            # Encoder returns feature map (B, 384, D/16, H/16, W/16)
            spatial_features = self.backbone(x)
            
        else:
            # Generic Fallback
            spatial_features = self.backbone(x)

        wants_spatial = getattr(self.head, 'accepts_spatial_features', False)
        
        if self.inference_mode == 'segmentation' or wants_spatial:
            return spatial_features
        
        pooled = self.pool_3d(spatial_features).flatten(1)
        # Ensure shape is (B, C)
        if pooled.ndim == 1: 
            pooled = pooled.unsqueeze(0)
            
        return pooled

    def forward(
        self, 
        x: torch.Tensor, 
        x_pair: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None, 
        mask_pair: Optional[torch.Tensor] = None, 
        aggregation: str = 'max'
    ):
        """
        Siamese Forward Pass with MIL Aggregation.
        
        Rationale:
        In datasets where side-specific labels are unavailable (only patient-level diagnosis),
        we treat the patient as a 'bag' of two instances (Left/Right TMJ).
        - 'max': Assumes 'pathology' class is dominant (if one side is sick, patient is sick).
        - 'mean': Assumes diffuse/bilateral condition or reduces noise for regression tasks.
        Args:
            x: Tensor for Left TMJ (Already flipped to Right anatomical space).
            x_pair: Tensor for Right TMJ (Optional).
            mask: (Optional) Boolean mask for Left TMJ background.
            mask_pair: (Optional) Boolean mask for Right TMJ background.
            aggregation: 'max' or 'mean' to combine bilateral features.
        """
        emb_L = self.get_embedding(x, mask=mask)

        # Case A: Joint-level Analysis (Independent processing)
        # Input batch is already stacked (2*B) or single side
        if x_pair is None:
            return self.head(emb_L)
        
        emb_R = self.get_embedding(x_pair, mask=mask_pair)
        
        # Aggregation Strategy
        if aggregation == 'max':
            patient_emb = torch.max(emb_L, emb_R)
        elif aggregation == 'mean':
            patient_emb = torch.mean(torch.stack([emb_L, emb_R]), dim=0)
        else:
            raise ValueError(f"Unknown aggregation strategy: {aggregation}")
            
        # Pass through the Task Head
        return self.head(patient_emb)

class MRIAnalysisHead(nn.Module):
    """
    Multi-Task Head for Joint-Level MRI Analysis.
    
    Structure:
    - Shared Bottleneck: Consolidates backbone features.
    - Task-Specific Heads:
        1. Bone Structure (Binary)
        2. Disc Position (Binary)
        3. Inflammation (Binary)
        4. Joint Space (Regression)
    """
    def __init__(self, in_features: int):
        super().__init__()
        
        self.fc_shared = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        # Independent projections
        # Output: [B, 1] raw logits (or scalar value)
        self.head_bone = nn.Linear(256, 1)
        self.head_disc = nn.Linear(256, 1)
        self.head_inflam = nn.Linear(256, 1)
        self.head_space = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shared = self.fc_shared(x)
        
        return torch.cat([
            self.head_bone(x_shared),   # Idx 0
            self.head_disc(x_shared),   # Idx 1
            self.head_inflam(x_shared), # Idx 2
            self.head_space(x_shared)   # Idx 3
        ], dim=1)