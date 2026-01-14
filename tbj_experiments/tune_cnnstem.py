"""
BertsWinMAE Stem Tuning Script
=========================================

This script implements the fine-tuning procedure for the BertsWinMAE architecture,
specifically targeting the 3D CNN Stem adaptation for volumetric inference.

Scientific Context:
-------------------
During pre-training (MAE), the model processes data in independent patches to facilitate
random masking. However, for downstream tasks and efficient inference, processing the
entire volume continuously (Volumetric Mode) is preferred to avoid boundary artifacts
and reduce computational overhead. This script fine-tunes the CNN Stem to bridge the
domain gap between patch-based training and volumetric inference.

Methodology:
------------
1.  Load pre-trained BertsWinMAE.
2.  Freeze the Transformer Encoder and Decoder.
3.  Unlock the Convolutional Stem (`patch_embed`).
4.  Optimize using Gradient Conductor (GCond) with a Lion-like update trajectory.
5.  Validate against a patch-based inference baseline.

"""

import os
import sys
import logging
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Iterator, List
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast
from tqdm import tqdm

# ==============================================================================
# 1. Environment & Imports Setup
# ==============================================================================

def setup_environment_paths():
    """Adds necessary project directories to sys.path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, '../..'))
    parent_dir_up = os.path.abspath(os.path.join(current_dir, '..'))
    
    paths_to_add = [parent_dir, parent_dir_up, current_dir]
    for p in paths_to_add:
        if p not in sys.path:
            sys.path.append(p)

setup_environment_paths()

try:
    # GCond: Gradient Conductor for stabilized optimization
    from GCond.grad_conductor import GradientConductor

    # Model Definition
    from bertswin_mae import BertsWinMAE, patchify_image

    # Training Utilities & Losses
    from train_maeswin import (
        Config as BaseConfig,
        setup_ddp, set_seed, setup_logging, atomic_save, reduce_tensor,
        prepare_dataloaders, process_batch_on_gpu,
        PhysicsInformedMAELoss, L2MAELoss,
        HounsfieldUnitInterpolator, CTNoiseSimulatorGPU,
        log_mae_visualizations_to_tensorboard
    )
except ImportError as e:
    logging.error(f"Critical Import Error: {e}")
    sys.exit(1)


# ==============================================================================
# 2. Configuration
# ==============================================================================

class StemTuningConfig(BaseConfig):
    """
    Configuration for Stem Fine-Tuning.
    Inherits from the base training configuration.
    """
    # Experiment Identifiers
    RUN_NAME: str = 'bertswin_adam_l2'
    PRETRAINED_CHECKPOINT_PATH: str = '/med_data/cbct-ct/dntum_mae/checkpoints_bertswin_adam_l2/best_model.pth'
    
    # Paths
    EXPERIMENTS_DIR: Path = Path('/med_data/cbct-ct/dntum_mae_tuned/')
    CHECKPOINT_DIR: Path = EXPERIMENTS_DIR / Path('checkpoints_' + RUN_NAME)
    LOG_DIR: Path = EXPERIMENTS_DIR / Path('logs_' + RUN_NAME)

    # Optimization Hyperparameters
    GPU_BATCH_SIZE_TRAIN: int = 4
    GPU_BATCH_SIZE_VAL: int = 5
    ACCUMULATION_STEPS: int = 25
    LEARNING_RATE: float = 1.5e-5
    
    # Training Loop Settings
    EPOCHS: int = 50
    TRAIN_STEPS_PER_EPOCH: int = 200
    MAE_MASK_RATIO: float = 0.0  # No masking during stem tuning
    
    # Loss Configuration
    TRAIN_LOSS_TYPE: str = 'custom_ssim'
    VALIDATION_PRIMARY_LOSS: str = 'custom_ssim_final_loss'

    # Gradient Conductor (GCond) Settings
    CONDUCTOR_USE_LION: bool = True
    CONDUCTOR_MOMENTUM_BETA: float = 0.9
    CONDUCTOR_TR_COEF: float = LEARNING_RATE
    CONDUCTOR_TR_CLIP: float = 10.0
    CONDUCTOR_PROJ_MAX_ITERS: int = 0
    CONDUCTOR_NORM_CAP: Optional[float] = None
    CONDUCTOR_DOMINANCE_WINDOW: int = 0
    CONDUCTOR_CONFLICT_THR: Tuple[float, float, float] = (-0.8, -0.5, 0.0)
    CONDUCTOR_NORM_EMA_BETA: float = 0.95
    CONDUCTOR_TB_WEIGHTS: Tuple[float, float] = (0.8, 0.2)
    CONDUCTOR_REMAP_POWER: float = 2.0
    CONDUCTOR_USE_SMOOTH_LOGIC: bool = True
    CONDUCTOR_STOCHASTIC: bool = True


# ==============================================================================
# 3. Model Wrappers (The Core Logic)
# ==============================================================================

class PatchBasedInferenceWrapper(nn.Module):
    """
    Wrapper for Baseline Validation.
    
    This class emulates the training-time behavior where 3D volumes are sliced 
    into patches *before* entering the CNN stem. Each patch is processed 
    independently, ensuring strict adherence to the pre-training protocol but 
    incurring high computational overhead.
    
    Flow:
        Input (B, 1, D, H, W) -> Patchify -> (B*N, 1, P, P, P) -> Stem -> Tokens
    """
    def __init__(self, original_model: BertsWinMAE):
        super().__init__()
        self.model = original_model
        self.patch_size = original_model.patch_size
        self.in_chans = original_model.in_chans

    def forward(self, x_vol: torch.Tensor, valid_for_masking_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # 1. Patchify: (B, 1, D, H, W) -> (B, N, P*P*P)
        patches_raw = patchify_image(x_vol, self.patch_size)
        B, N, _ = patches_raw.shape
        
        # 2. Reshape for Isolated Stem Processing: (B*N, 1, P, P, P)
        # Crucial: This breaks spatial continuity between patches in the stem
        x_patches_spatial = patches_raw.view(
            B * N, self.in_chans, self.patch_size, self.patch_size, self.patch_size
        )
        
        # 3. Forward Stem: (B*N, Embed, 1, 1, 1)
        x_embedded = self.model.patch_embed(x_patches_spatial)
        
        # 4. Reshape to Sequence: (B, N, Embed)
        x_tokens = x_embedded.view(B, N, self.model.encoder_embed_dim)
        
        # 5. Add Positional Embeddings
        x_tokens = x_tokens + self.model.pos_embed
        
        # 6. Swin Transformer Encoder
        Dg, Hg, Wg = self.model.grid_size
        x_swin_in = x_tokens.view(B, Dg, Hg, Wg, -1)
        encoded_features = self.model.features(x_swin_in)
        
        # 7. Decoder
        encoded_spatial = encoded_features.permute(0, 4, 1, 2, 3).contiguous()
        _, recon_cube = self.model.forward_decoder(encoded_spatial)
        
        # 8. Auxiliary Outputs
        recon_patches = patchify_image(recon_cube, self.patch_size)
        dummy_mask = torch.ones((B, N), device=x_vol.device, dtype=torch.float32)

        return {
            'reconstructed_cube': recon_cube,
            'reconstructed_patches': recon_patches,
            'mae_mask_1d': dummy_mask
        }


class VolumetricBertsWinWrapper(nn.Module):
    """
    Wrapper for Volumetric Inference (The Tuning Target).
    
    This class implements the desired inference behavior: the entire 3D volume
    is passed directly to the CNN Stem. This allows for fully convolutional 
    processing, maintaining global context and reducing kernel launch overhead.
    
    Flow:
        Input (B, 1, D, H, W) -> Stem -> (B, Embed, D/P, H/P, W/P) -> Tokens
    """
    def __init__(self, original_model: BertsWinMAE):
        super().__init__()
        self.model = original_model
        self.patch_size = original_model.patch_size
        
    def forward(self, x_vol: torch.Tensor, valid_for_masking_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # 1. Global Convolutional Stem
        # Input: (B, C, D, H, W) -> Output: (B, Embed, D/P, H/P, W/P)
        x_stem = self.model.patch_embed(x_vol) 
        
        # 2. Prepare for Transformer
        x_spatial = x_stem.permute(0, 2, 3, 4, 1).contiguous()
        B, Dg, Hg, Wg, C = x_spatial.shape
        x_tokens = x_spatial.view(B, -1, C)
        
        # 3. Add Positional Embeddings
        x_tokens = x_tokens + self.model.pos_embed 
        
        # 4. Swin Transformer Encoder
        x_swin_in = x_tokens.view(B, Dg, Hg, Wg, C)
        encoded_features = self.model.features(x_swin_in) 
        
        # 5. Decoder
        encoded_spatial = encoded_features.permute(0, 4, 1, 2, 3).contiguous()
        _, recon_cube = self.model.forward_decoder(encoded_spatial)
        
        recon_patches = patchify_image(recon_cube, self.patch_size)
        dummy_mask = torch.ones((B, recon_patches.shape[1]), device=recon_patches.device, dtype=torch.float32)

        return {
            'reconstructed_cube': recon_cube,
            'reconstructed_patches': recon_patches,
            'mae_mask_1d': dummy_mask 
        }


# ==============================================================================
# 4. Utility Functions
# ==============================================================================

def prepare_model_for_stem_tuning(model: BertsWinMAE) -> None:
    """
    Freezes the Transformer and Decoder, enabling gradients only for the Stem.
    
    Args:
        model: The base BertsWinMAE model.
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze only the Patch Embedding (Stem) layers
    for param in model.patch_embed.parameters():
        param.requires_grad = True
        
    logging.info("Model Frozen. Optimization Target: 'model.patch_embed' only.")

def clean_state_dict(state_dict: Dict[str, Any]) -> OrderedDict:
    """
    Removes DDP/TorchCompile prefixes from state dict keys for strict loading.
    
    Args:
        state_dict: Raw state dictionary from checkpoint.
    Returns:
        Cleaned OrderedDict.
    """
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "").replace("_orig_mod.", "")
        new_sd[name] = v
    return new_sd

def print_metrics_table(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """Prints a formatted ASCII table of metrics."""
    logging.info(f"┌{'─'*60}┐")
    logging.info(f"│ {title:<58} │")
    logging.info(f"├{'─'*40}┬{'─'*19}┤")
    sorted_keys = sorted(metrics.keys())
    for k in sorted_keys:
        val = metrics[k]
        logging.info(f"│ {k:<38} │ {val:17.6f} │")
    logging.info(f"└{'─'*60}┘")


# ==============================================================================
# 5. Engine: Training & Validation
# ==============================================================================

@torch.no_grad()
def run_validation_epoch(
    model: nn.Module, 
    val_loader: Any, 
    loss_fns: Dict[str, nn.Module], 
    cfg: StemTuningConfig, 
    device: torch.device, 
    epoch: int, 
    rank: int, 
    world_size: int, 
    writer: Optional[SummaryWriter], 
    prefix: str = "Val"
) -> Dict[str, float]:
    """
    Executes a full validation epoch. Supports both Patch-based and Volumetric wrappers.
    """
    model.eval()
    total_losses_sum = defaultdict(lambda: torch.tensor(0.0, device=device))
    valid_batches_count = torch.tensor(0.0, device=device)
    
    pbar_desc = f"[{prefix}] Epoch {epoch}"
    iterator = tqdm(val_loader, desc=pbar_desc, leave=False) if rank == 0 else val_loader

    for step, batch in enumerate(iterator):
        if batch is None: continue
        valid_batches_count += 1
        
        # Move data to GPU
        for k, v in batch.items():
            if isinstance(v, torch.Tensor): 
                batch[k] = v.to(device, non_blocking=True)
            
        processed_batch = process_batch_on_gpu(batch, cfg, epoch)
        
        with autocast('cuda', dtype=torch.bfloat16):
            model_output = model(processed_batch['input_cube'], valid_for_masking_mask=None)

            # Compute Losses
            for loss_name, loss_fn in loss_fns.items():
                loss_value, loss_components = loss_fn(model_output, processed_batch)
                for key, value in loss_components.items():
                    total_losses_sum[f'{loss_name}_{key}'] += value.detach()
                total_losses_sum[f'{loss_name}_final_loss'] += loss_value.detach()

        # Log visualizations for the first batch
        if step == 0 and rank == 0 and writer:
             log_mae_visualizations_to_tensorboard(
                writer=writer,
                originals=processed_batch['input_cube'],
                reconstructions=model_output['reconstructed_cube'],
                mae_mask_1d=model_output['mae_mask_1d'],
                cfg=cfg,
                epoch=epoch,
                num_samples_to_log=min(4, processed_batch['input_cube'].shape[0])
            )

    # Sync Metrics across GPUs
    if world_size > 1:
        sorted_keys = sorted(total_losses_sum.keys())
        tensor_list = [total_losses_sum[k] for k in sorted_keys]
        if tensor_list:
            all_losses = torch.stack(tensor_list)
            torch.distributed.all_reduce(all_losses)
            total_losses_sum = {k: all_losses[i] for i, k in enumerate(sorted_keys)}
        torch.distributed.all_reduce(valid_batches_count)

    if valid_batches_count.item() == 0:
        return {}

    avg_metrics = {k: (v / valid_batches_count.item()).item() for k, v in total_losses_sum.items()}
    return avg_metrics


def train_one_epoch_volumetric(
    model: nn.Module, 
    train_loader: Any, 
    optimizer: torch.optim.Optimizer, 
    lr_optimizer: torch.optim.Optimizer, 
    conductor: GradientConductor, 
    lr_scheduler: Any, 
    cfg: StemTuningConfig, 
    device: torch.device, 
    epoch: int, 
    rank: int, 
    world_size: int, 
    writer: Optional[SummaryWriter], 
    hu_augmenter: Any, 
    noise_simulator: Any, 
    train_iter_state: Dict[str, Any]
) -> Dict[str, float]:
    """
    Executes one training epoch using the Gradient Conductor for physics-constrained tuning.
    Includes custom iterator logic to handle epoch boundaries seamlessly.
    """
    model.train()
    epoch_losses = {'final_loss': 0.0}
    num_steps = cfg.TRAIN_STEPS_PER_EPOCH
    
    pbar = tqdm(range(num_steps), desc=f"Epoch {epoch} [Tune]", disable=(rank != 0), leave=False)

    # Infinite Iterator Logic for Fixed Step Counts
    def get_next_batch():
        nonlocal train_iter_state
        try:
            batch = next(train_iter_state['iterator'])
        except StopIteration:
            train_iter_state['real_epoch'] += 1
            train_loader.sampler.set_epoch(train_iter_state['real_epoch'])
            train_iter_state['iterator'] = iter(train_loader)
            batch = next(train_iter_state['iterator'])
        
        # Recursive retry if batch is None
        if batch is None: return get_next_batch()
        return batch

    for step in pbar:
        current_lr = lr_optimizer.param_groups[0]['lr']
        conductor.trust_ratio_coef = current_lr

        # Closure for GCond
        def data_provider():
            batch = get_next_batch()
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): 
                    batch[k] = v.to(device, non_blocking=True)
            
            # Apply Augmentations (HU + Noise)
            processed_batch = process_batch_on_gpu(batch, cfg, epoch, hu_augmenter, noise_simulator)
            
            # Args for model.forward, and context for loss
            return (
                {'args': (processed_batch['input_cube'],), 'kwargs': {'valid_for_masking_mask': None}}, 
                processed_batch
            )

        # Optimization Step via Gradient Conductor
        stats = conductor.step(data_provider=data_provider)
        
        # Apply Updates
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP_VAL)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        
        # Logging
        loss_val = stats.get('loss/physics_loss', 0.0)
        loss_tensor = torch.tensor(loss_val, device=device)
        if world_size > 1: 
            loss_tensor = reduce_tensor(loss_tensor)
        
        avg_loss_step = loss_tensor.item()
        epoch_losses['final_loss'] += avg_loss_step

        if rank == 0:
            pbar.set_postfix(loss=avg_loss_step, lr=f"{current_lr:.2e}", grad=f"{total_norm:.2e}")
            if writer and (step % 20 == 0):
                gs = epoch * num_steps + step
                writer.add_scalar('tune/train_loss', avg_loss_step, gs)
                writer.add_scalar('tune/lr', current_lr, gs)
                writer.add_scalar('tune/grad', total_norm, gs)

    return {'final_loss': epoch_losses['final_loss'] / num_steps}


# ==============================================================================
# 6. Main Execution
# ==============================================================================

def main():
    cfg = StemTuningConfig()
    rank, world_size = setup_ddp()
    set_seed(cfg.SEED, rank)
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
    setup_logging(rank, cfg.LOG_DIR)
    
    if rank == 0:
        cfg.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        writer = SummaryWriter(log_dir=str(cfg.LOG_DIR / "tensorboard"))
        logging.info(f"LAUNCHING STEM TUNING (GCond) | Base Checkpoint: {cfg.PRETRAINED_CHECKPOINT_PATH}")
    else:
        writer = None

    # 1. Initialize Base Model
    original_model = BertsWinMAE(
        img_size=cfg.INPUT_CUBE_SIZE,
        patch_size=cfg.EFFECTIVE_PATCH_SIZE,
        in_chans=1,
        encoder_embed_dim=cfg.ENCODER_EMBED_DIM,
        encoder_depths=cfg.SWIN_DEPTHS,
        encoder_num_heads=cfg.SWIN_NUM_HEADS,
        swin_window_size=cfg.SWIN_WINDOW_SIZE,
        decoder_embed_dim=cfg.DECODER_EMBED_DIM,
        stem_base_dim=cfg.STEM_BASE_DIM,
        mask_ratio=0.0
    ).to(device)

    # 2. Load Pre-trained Weights
    if not os.path.exists(cfg.PRETRAINED_CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.PRETRAINED_CHECKPOINT_PATH}")
    
    checkpoint = torch.load(cfg.PRETRAINED_CHECKPOINT_PATH, map_location=device, weights_only=False)
    sd = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    original_model.load_state_dict(clean_state_dict(sd), strict=True)
    
    # 3. Data Loading
    train_loader, val_loader, train_ids, val_ids = prepare_dataloaders(rank, world_size, cfg, checkpoint=None)
    train_iter_state = {'iterator': iter(train_loader), 'real_epoch': 0}
    
    # 4. Loss Functions
    loss_factory = {
        'custom_ssim': PhysicsInformedMAELoss(cfg).to(device),
        'l2': L2MAELoss(cfg).to(device)
    }
    loss_fn_train = loss_factory[cfg.TRAIN_LOSS_TYPE]

    # ==================================================================
    # 5. TARGET BASELINE: Validation in Patch-based Mode
    # ==================================================================
    if rank == 0: 
        logging.info("TARGET CHECK: Validating Original Stem in Patch-based Mode...")
    
    patch_wrapper = PatchBasedInferenceWrapper(original_model).to(device)
    
    patch_metrics = run_validation_epoch(
        patch_wrapper, val_loader, loss_factory, cfg, device, -1, rank, world_size, writer, prefix="Patch_Target"
    )
    
    if rank == 0:
        print_metrics_table(patch_metrics, title="Patch-Based Target Metrics")
        if writer:
            for k, v in patch_metrics.items():
                writer.add_scalar(f'Target_Metrics/{k}', v, 0)

    # ==================================================================
    # 6. VOLUMETRIC BASELINE: Validation in Volumetric Mode (Before Tuning)
    # ==================================================================
    if rank == 0: 
        logging.info("VOLUMETRIC CHECK: Validating Original Stem in Volumetric Mode (Untuned)...")
    
    # Wrap for Volumetric Tuning
    model = VolumetricBertsWinWrapper(original_model).to(device)
    model = torch.compile(model)
    
    vol_metrics = run_validation_epoch(
        model, val_loader, loss_factory, cfg, device, 0, rank, world_size, writer, prefix="Volumetric_Baseline"
    )
    
    if rank == 0:
        print_metrics_table(vol_metrics, title="Volumetric Baseline (Before Tuning)")
        if writer:
            for k, v in vol_metrics.items():
                writer.add_scalar(f'Validation/{k}', v, 0)

    # ==================================================================
    # 7. TRAINING SETUP
    # ==================================================================
    
    # Prepare Model: Freeze
    prepare_model_for_stem_tuning(model.model)

    if world_size > 1:
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=False)

    # Optimizers & Gradient Conductor
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1.0)
    lr_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1, device=device))], lr=cfg.LEARNING_RATE)

    conductor = GradientConductor(
        model=model,
        loss_fns={'physics_loss': lambda out, batch: loss_fn_train(out, batch)[0]},
        lambdas={'physics_loss': 1.0},
        accumulation_steps=cfg.ACCUMULATION_STEPS,
        momentum_beta=cfg.CONDUCTOR_MOMENTUM_BETA,
        use_lion=cfg.CONDUCTOR_USE_LION,
        trust_ratio_coef=cfg.CONDUCTOR_TR_COEF,
        trust_ratio_clip=cfg.CONDUCTOR_TR_CLIP,
        ddp_sync="avg",
        projection_max_iters=0,
        stochastic_accumulation=cfg.CONDUCTOR_STOCHASTIC
    )

    # Schedulers
    warmup_steps = 2 * cfg.TRAIN_STEPS_PER_EPOCH
    total_steps = cfg.EPOCHS * cfg.TRAIN_STEPS_PER_EPOCH
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(lr_optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lr_optimizer, T_max=(total_steps - warmup_steps), eta_min=cfg.COSINE_ETA_MIN)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(lr_optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])

    # Goal: Beat or match the patch-based metrics
    best_loss = patch_metrics.get(cfg.VALIDATION_PRIMARY_LOSS, float('inf')) 
    if rank == 0: 
        logging.info(f"Optimization Goal (Patch-based Loss): {best_loss:.6f}")

    # ==================================================================
    # 8. OPTIMIZATION LOOP
    # ==================================================================
    for epoch in range(1, cfg.EPOCHS + 1):
        if world_size > 1: 
            val_loader.sampler.set_epoch(epoch)
            
        train_metrics = train_one_epoch_volumetric(
            model, train_loader, optimizer, lr_optimizer, conductor, lr_scheduler, 
            cfg, device, epoch, rank, world_size, writer, None, None, train_iter_state
        )
        
        val_metrics = run_validation_epoch(
            model, val_loader, loss_factory, cfg, device, epoch, rank, world_size, writer, prefix="Val"
        )
        
        main_val_loss = val_metrics[cfg.VALIDATION_PRIMARY_LOSS]
        
        if rank == 0:
            print_metrics_table(val_metrics, title=f"Validation Metrics (Epoch {epoch})")
            if writer:
                for k, v in val_metrics.items():
                    writer.add_scalar(f'Validation/{k}', v, epoch)

            # Save if we improve towards the Patch-based baseline
            if main_val_loss < best_loss:
                best_loss = main_val_loss
                logging.info(f"NEW BEST! {cfg.VALIDATION_PRIMARY_LOSS}: {best_loss:.6f}")
                
                checkpoint_data = {
                    'epoch': epoch,
                    'real_epoch': train_iter_state['real_epoch'],
                    'model_state_dict': original_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_optimizer_state_dict': lr_optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'conductor_state_dict': conductor.state_dict(),
                    'best_val_loss': best_loss,
                    'train_patient_ids': train_ids,
                    'val_patient_ids': val_ids,
                }
                atomic_save(checkpoint_data, cfg.CHECKPOINT_DIR / "best_cnnstem_tuned.pth")

if __name__ == '__main__':
    main()