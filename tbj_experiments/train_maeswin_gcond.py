import os
import sys
import h5py
import hdf5plugin
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import logging
import shutil
import math
import json
import random
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any, List, Callable
from collections import defaultdict

from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation

import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from torchvision.models.video.swin_transformer import SwinTransformerBlock, ShiftedWindowAttention3d
from torchvision.models.swin_transformer import PatchMerging 
from functools import partial

#from torch.profiler import profile, record_function, ProfilerActivity

try:
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from GCond.grad_conductor import GradientConductor
except ImportError:
    logging.error("GradientConductor not found. Aborting.")
    sys.exit(1)

try:
    from train_maeswin import (
        Config as BaseConfig,
        setup_ddp,
        set_seed,
        setup_logging,
        log_config,
        atomic_save,
        EarlyStopping,
        reduce_tensor,
        HounsfieldUnitInterpolator,
        CTNoiseSimulatorGPU,
        prepare_dataloaders,
        HybridMAE,
        L2MAELoss,
        SubpatchSSIMLoss,
        PhysicsInformedMAELoss,
        process_batch_on_gpu,
        log_mae_visualizations_to_tensorboard,
        validate
    )
except ImportError:
    logging.error("Failed to import 'train_maeswin.py'. Ensure it is located in the same directory.")
    sys.exit(1)

os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

#pip install hdf5plugin matplotlib tensorboard scipy
#torchrun --nproc_per_node 4 train_maeswin_gcond.py

class Config(BaseConfig):
    """
    Configuration class defining hyperparameters and system settings.
    """
    # --- Paths ---
    DATA_HDF5_PATH = './tbj_cbct.hdf5'
    EXPERIMENTS_DIR = Path('/workspace/experiments/')
    RUN_NAME = 'bertswin_gcond_ssim_a'
    CHECKPOINT_DIR = EXPERIMENTS_DIR / Path('checkpoints_' + RUN_NAME)
    LOG_DIR = EXPERIMENTS_DIR / Path('logs_' + RUN_NAME)
    
    SEED = 42
    
    LEARNING_RATE = 1.5e-5
    EPOCHS = 1000
    TRAIN_STEPS_PER_EPOCH = 300

    USE_FOV_AWARE_MASKING = True

    ACCUMULATION_STEPS = 5
    GRADIENT_CLIP_VAL = 10.0
    GPU_BATCH_SIZE_TRAIN = 10
    GPU_BATCH_SIZE_VAL = 6
    NUM_WORKERS_TRAIN = 6
    NUM_WORKERS_VAL = 4
    
    TRAIN_LOSS_TYPE = 'ssim_a' # ssim_a, l2, custom_ssim
    VALIDATION_PRIMARY_LOSS = 'ssim_a_final_loss' # ssim_a_final_loss, custom_ssim_final_loss
    PHY_LOSS_WEIGHTS = (0.3, 0.5, 0.2)
    LCS_LOSS_WEIGHTS = (0.3, 0.2, 0.5) # br, cntr, str
    LOSS_SUB_PATCH_SIZE = 8

    CONDUCTOR_USE_LION = True
    CONDUCTOR_MOMENTUM_BETA = 0.9
    CONDUCTOR_TR_COEF = LEARNING_RATE 
    CONDUCTOR_TR_CLIP = 10.0
    
    CONDUCTOR_PROJ_MAX_ITERS = 3
    CONDUCTOR_NORM_CAP = None
    CONDUCTOR_DOMINANCE_WINDOW = 0
    CONDUCTOR_CONFLICT_THR = (-0.8, -0.5, 0.0) 
    CONDUCTOR_NORM_EMA_BETA = 0.95
    CONDUCTOR_TB_WEIGHTS = (0.8, 0.2)
    CONDUCTOR_REMAP_POWER = 2.0
    CONDUCTOR_USE_SMOOTH_LOGIC = True
    CONDUCTOR_STOCHASTIC = True

    WARMUP_EPOCHS = 20
    COSINE_ETA_MIN = LEARNING_RATE / 10
    PATIENCE_EPOCHS = 50
    SCHEDULER_TYPE = 'Cosine'

# =================================================================================
# Training Loop
# =================================================================================

def train_one_epoch(model: nn.Module, train_loader: DataLoader, 
                    optimizer: torch.optim.Optimizer, 
                    lr_optimizer: torch.optim.Optimizer,
                    conductor: GradientConductor, lr_scheduler, loss_fn: nn.Module, cfg: Config, 
                    device: torch.device, epoch: int, rank: int, world_size: int, writer: Optional[SummaryWriter],
                    hu_augmenter: Optional[nn.Module], noise_simulator: Optional[nn.Module],
                    train_iter_state: Dict[str, Any],
                ) -> Dict[str, float]:
    """
    Executes a single training epoch with gradient accumulation and DDP support.
    
    Uses a virtual epoch approach where 'num_steps_per_epoch' defines the epoch length,
    handling the infinite data stream via 'train_iter_state'.
    """
    model.train()
    
    epoch_losses = {'final_loss': 0.0}
    num_steps_per_epoch = cfg.TRAIN_STEPS_PER_EPOCH 
    
    pbar = tqdm(range(num_steps_per_epoch), desc=f"Epoch {epoch} [Train]", disable=(rank != 0), leave=False) 

    def get_next_batch():
        nonlocal train_iter_state 
        try:
            batch = next(train_iter_state['iterator'])
        except StopIteration:
            train_iter_state['real_epoch'] += 1
            train_loader.sampler.set_epoch(train_iter_state['real_epoch'])
            train_iter_state['iterator'] = iter(train_loader)
            batch = next(train_iter_state['iterator'])
        
        if batch is None:
            logging.warning(f"Epoch {epoch}: Skipped empty/corrupt batch.")
            return get_next_batch() 
        
        return batch
    
    for step in pbar:
        #if (step > 20):
        #    break
        current_lr = lr_optimizer.param_groups[0]['lr']
        conductor.trust_ratio_coef = current_lr

        def data_provider():
            batch = get_next_batch()
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device, non_blocking=True)
            
            processed_batch = process_batch_on_gpu(
                batch, cfg, epoch, hu_augmenter, noise_simulator
            )
            
            model_x_args = (processed_batch['input_cube'],)
            model_x_kwargs = {'valid_for_masking_mask': processed_batch['valid_for_masking_mask']}
            
            loss_y = processed_batch 
            
            return {'args': model_x_args, 'kwargs': model_x_kwargs}, loss_y

        stats = conductor.step(data_provider=data_provider)
        
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.GRADIENT_CLIP_VAL
        )
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        
        loss_tensor = torch.tensor(stats.get('loss/physics_loss', 0.0), device=device)
        if world_size > 1:
            loss_tensor = reduce_tensor(loss_tensor)
        
        avg_loss_step = loss_tensor.item()
        epoch_losses['final_loss'] += avg_loss_step 

        if rank == 0:
            pbar_loss = stats.get('loss/physics_loss', 0.0)
            pbar.set_postfix(loss=pbar_loss, lr=f"{current_lr:.2e}")
            
            if writer and (step % 20 == 0): 
                global_step = epoch * num_steps_per_epoch + step
                writer.add_scalar('train_step/learning_rate', current_lr, global_step)

                for k, v in stats.items(): 
                    if k == 'step':
                        continue
                    if k == 'loss/physics_loss':
                        writer.add_scalar('train_step/total_loss', v, global_step)
                    elif k == 'raw_norm/physics_loss':
                        writer.add_scalar('train_step/grad_norm', v, global_step)
                    else:
                        safe_key = k.replace('/', '_')
                        writer.add_scalar(f'conductor/{safe_key}', v, global_step)
                
                writer.add_scalar('conductor/grad_norm_clipped', total_norm.item(), global_step)
    
    avg_epoch_losses = {k: v / num_steps_per_epoch for k, v in epoch_losses.items()}
    return avg_epoch_losses


@torch.no_grad()
def validate(model: nn.Module, val_loader: DataLoader, 
             validation_loss_fns: Dict[str, nn.Module],
             cfg: Config, 
             device: torch.device, epoch: int, rank: int, world_size: int, writer: Optional[SummaryWriter]) -> Dict[str, float]:
    """
    Performs a single validation pass.
    """
    model.eval()
    
    total_losses_sum = defaultdict(lambda: torch.tensor(0.0, device=device))
    
    valid_batches_count = torch.tensor(0.0, device=device)
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", disable=(rank != 0), leave=False)
    
    for step, batch in enumerate(pbar):
        
        if batch is None:
            logging.warning(f"Epoch {epoch} [Val]: Skipped empty/corrupt batch.")
            continue
            
        valid_batches_count += 1
            
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(device, non_blocking=True)
                
        processed_batch = process_batch_on_gpu(
            batch, cfg, epoch, hu_augmenter=None, noise_simulator=None
        )
        
        with autocast('cuda', dtype=torch.bfloat16):
            model_output = model(
                processed_batch['input_cube'],
                valid_for_masking_mask=processed_batch['valid_for_masking_mask']
            )

            for loss_name, loss_fn in validation_loss_fns.items():
                loss_value, loss_components = loss_fn(model_output, processed_batch)
                
                for key, value in loss_components.items():
                    total_losses_sum[f'{loss_name}_{key}'] += value.detach()

                total_losses_sum[f'{loss_name}_final_loss'] += loss_value.detach()
        
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
            
    
    if world_size > 1:
        sorted_keys = sorted(total_losses_sum.keys())
        tensor_list_sorted = [total_losses_sum[k] for k in sorted_keys]
        
        if not tensor_list_sorted:
             pass
        else:
            all_losses_tensor = torch.stack(tensor_list_sorted)
            dist.all_reduce(all_losses_tensor, op=dist.ReduceOp.SUM)
            
            total_losses_sum = {
                key: all_losses_tensor[i] for i, key in enumerate(sorted_keys)
            }
        dist.all_reduce(valid_batches_count, op=dist.ReduceOp.SUM)

    if valid_batches_count.item() == 0:
        logging.warning(f"Epoch {epoch} [Val]: No valid batches for evaluation")
        return {k: 0.0 for k in total_losses_sum.keys()}

    avg_losses_tensors = {k: v / valid_batches_count.item() for k, v in total_losses_sum.items()}
     
    avg_val_losses = {key: val.item() for key, val in avg_losses_tensors.items()}
     
    return avg_val_losses

# =================================================================================
# Main Function
# =================================================================================

def main():
    cfg = Config()
    rank, world_size = setup_ddp()
    set_seed(cfg.SEED, rank)
    
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
    setup_logging(rank, cfg.LOG_DIR)
    if rank == 0:
        cfg.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        writer = SummaryWriter(log_dir=str(cfg.LOG_DIR / "tensorboard"))
        log_config(cfg)
    else:
        writer = None

    latest_checkpoint_path = cfg.CHECKPOINT_DIR / "latest_checkpoint.pth"
    checkpoint = None
    if latest_checkpoint_path.exists():
        checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        if rank == 0: logging.info(f"Resuming from epoch {checkpoint['epoch']}")
    
    train_loader, val_loader, train_ids, val_ids = prepare_dataloaders(rank, world_size, cfg, checkpoint)
    
    model = HybridMAE(cfg).to(device)
    model.initialize_weights()
    model = torch.compile(model)
    if world_size > 1:
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    loss_factory = {
        'custom_ssim': PhysicsInformedMAELoss(cfg).to(device),
        'l2': L2MAELoss(cfg).to(device),
        'ssim_a': SubpatchSSIMLoss(cfg).to(device)
    }

    if cfg.TRAIN_LOSS_TYPE not in loss_factory:
        raise ValueError(
            f"Unknown loss type: {cfg.TRAIN_LOSS_TYPE}. "
            f"Available: {list(loss_factory.keys())}"
        )
    primary_loss_prefix = None
    for key in loss_factory.keys():
        if cfg.VALIDATION_PRIMARY_LOSS.startswith(key):
            primary_loss_prefix = key
            break

    if primary_loss_prefix is None:
        raise ValueError(
            f"Prefix '{primary_loss_prefix}' from VALIDATION_PRIMARY_LOSS ('{cfg.VALIDATION_PRIMARY_LOSS}') "
            f"not found in loss_factory. Available prefixes: {list(loss_factory.keys())}"
        )

    loss_fn_train = loss_factory[cfg.TRAIN_LOSS_TYPE]
    if rank == 0: 
        logging.info(f"Training with loss: {cfg.TRAIN_LOSS_TYPE}")
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=1.0, 
        momentum=0.0, 
        weight_decay=0.0
    )
    lr_optimizer = torch.optim.SGD(
        [torch.nn.Parameter(torch.zeros(1, device=device))], 
        lr=cfg.LEARNING_RATE
    )

    def conductor_loss_fn_wrapper(model_output_dict, processed_batch_dict):
        final_loss, _ = loss_fn_train(model_output_dict, processed_batch_dict)
        return final_loss

    conductor_losses = {'physics_loss': conductor_loss_fn_wrapper}
    conductor_lambdas = {'physics_loss': 1.0}

    conductor = GradientConductor(
        model=model, 
        loss_fns=conductor_losses,
        lambdas=conductor_lambdas,
        accumulation_steps=cfg.ACCUMULATION_STEPS,
        
        momentum_beta=cfg.CONDUCTOR_MOMENTUM_BETA,
        use_lion=cfg.CONDUCTOR_USE_LION,
        trust_ratio_coef=cfg.CONDUCTOR_TR_COEF,
        trust_ratio_clip=cfg.CONDUCTOR_TR_CLIP,
        
        ddp_sync="avg", 
        
        projection_max_iters=cfg.CONDUCTOR_PROJ_MAX_ITERS,
        norm_cap=cfg.CONDUCTOR_NORM_CAP,
        dominance_window=cfg.CONDUCTOR_DOMINANCE_WINDOW,
        conflict_thresholds=cfg.CONDUCTOR_CONFLICT_THR,
        norm_ema_beta=cfg.CONDUCTOR_NORM_EMA_BETA,
        tie_breaking_weights=cfg.CONDUCTOR_TB_WEIGHTS,
        remap_power=cfg.CONDUCTOR_REMAP_POWER,
        use_smooth_logic=cfg.CONDUCTOR_USE_SMOOTH_LOGIC,
        stochastic_accumulation=cfg.CONDUCTOR_STOCHASTIC
    )

    hu_augmenter = HounsfieldUnitInterpolator(cfg).to(device) if cfg.USE_HU_AUG else None
    noise_simulator = CTNoiseSimulatorGPU(cfg).to(device) if cfg.USE_NOISE_AUG else None
    if rank == 0:
        if hu_augmenter: logging.info("HU augmentation enabled.")
        if noise_simulator: logging.info("CT noise simulation enabled.")
    
    num_steps_per_epoch_virtual = cfg.TRAIN_STEPS_PER_EPOCH
    warmup_steps = cfg.WARMUP_EPOCHS * num_steps_per_epoch_virtual
    total_steps = cfg.EPOCHS * num_steps_per_epoch_virtual

    if cfg.SCHEDULER_TYPE == 'Cosine':
        if rank == 0:
            logging.info(f"Using CosineAnnealingLR: Warmup={cfg.WARMUP_EPOCHS} epochs ({warmup_steps} steps), T_max={total_steps - warmup_steps} steps")

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            lr_optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
        )
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            lr_optimizer, T_max=(total_steps - warmup_steps), eta_min=cfg.COSINE_ETA_MIN
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            lr_optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps]
        )

    elif cfg.SCHEDULER_TYPE == 'Linear':
        if rank == 0:
            logging.info(f"Using LinearLR (for testing): Warmup={cfg.WARMUP_EPOCHS} epochs ({warmup_steps} steps)")
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            lr_optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
        )
    else:
        raise ValueError(f"Unknown SCHEDULER_TYPE: {cfg.SCHEDULER_TYPE}")
    
    start_epoch = 0
    real_epoch_start = 0
    best_val_loss = float('inf')
    early_stopper = EarlyStopping(patience=cfg.PATIENCE_EPOCHS)
    
    if checkpoint:
        (model.module if world_size > 1 else model).load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'lr_optimizer_state_dict' in checkpoint:
            lr_optimizer.load_state_dict(checkpoint['lr_optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        real_epoch_start = checkpoint.get('real_epoch', start_epoch)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if 'early_stopper_state_dict' in checkpoint:
            early_stopper.load_state_dict(checkpoint['early_stopper_state_dict'])
        if 'conductor_state_dict' in checkpoint:
            conductor.load_state_dict(checkpoint['conductor_state_dict'])

    train_loader.sampler.set_epoch(real_epoch_start)
    
    train_iter_state = {
        'iterator': iter(train_loader),
        'real_epoch': real_epoch_start 
    }

    if rank == 0:
        logging.info("Starting training...")
        
    for epoch in range(start_epoch, cfg.EPOCHS):
        if world_size > 1:
            val_loader.sampler.set_epoch(epoch)
        
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, lr_optimizer, conductor, lr_scheduler, loss_fn_train, cfg, 
            device, epoch, rank, world_size, writer,
            hu_augmenter, noise_simulator,
            train_iter_state=train_iter_state
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        val_metrics = validate(
             model, val_loader, loss_factory, cfg, device, epoch, rank, world_size, writer
        )
        
        if rank == 0:
            main_train_loss = train_metrics['final_loss']
            main_val_loss = val_metrics[cfg.VALIDATION_PRIMARY_LOSS]
            
            log_msg = (
                f"Epoch {epoch:03d} | "
                f"Train Loss ({cfg.TRAIN_LOSS_TYPE}): {main_train_loss:.5f} | "
                f"Val Loss ({primary_loss_prefix}): {main_val_loss:.5f} | "
                f"LR: {lr_optimizer.param_groups[0]['lr']:.2e}"
            )
            logging.info(log_msg)
            if writer:
                for key, value in train_metrics.items():
                    writer.add_scalar(f'Train/{key}', value, epoch)
                for key, value in val_metrics.items():
                    writer.add_scalar(f'Validation/{key}', value, epoch)
               
                writer.add_scalar('Meta/Learning_Rate', lr_optimizer.param_groups[0]['lr'], epoch)

            
            is_best = main_val_loss < best_val_loss
            if is_best:
                best_val_loss = main_val_loss
            
            checkpoint_data = {
                'epoch': epoch,
                'real_epoch': train_iter_state['real_epoch'],
                'model_state_dict': (model.module if world_size > 1 else model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_optimizer_state_dict': lr_optimizer.state_dict(), 
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'conductor_state_dict': conductor.state_dict(),
                'best_val_loss': best_val_loss,
                'early_stopper_state_dict': early_stopper.state_dict(),
                'train_patient_ids': train_ids,
                'val_patient_ids': val_ids,
            }
            
            atomic_save(checkpoint_data, cfg.CHECKPOINT_DIR / "latest_checkpoint.pth")
            if is_best:
                logging.info(f"New best model saved with val loss: {best_val_loss:.6f}")
                atomic_save(checkpoint_data, cfg.CHECKPOINT_DIR / "best_model.pth")
                
            early_stopper(main_val_loss)
            if early_stopper.early_stop:
                logging.info("Early stopping. Training finished.")
                break

        if world_size > 1:
            dist.barrier()
            
    if rank == 0 and writer:
        writer.close()
        logging.info("Training completed.")

if __name__ == '__main__':
    main()