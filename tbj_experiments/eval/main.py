import argparse
import importlib.util
import sys
import os
import logging
import json
import random
import copy
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any
from functools import partial

# Internal modular imports
from configs import EvalConfig
from tasks import (
    TMJCoreTask, TMJSegmentationTask, worker_init_fn,
    TMJInflammationTask, TMJBoneTask, TMJDiscTask, TMJSpaceTask
)
from models import ModelFactory, TMJAnalysisModel, inject_lora_to_model
from engine import train_epoch, validate_epoch, EarlyStopping

try:
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from bertswin_mae import BertsWinMAE, patchify_image #BertsWinMAE.
except ImportError:
    raise ImportError("Warning: Failed to import BertsWinMAE")

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def setup_ddp():
    """
    Initializes the Distributed Data Parallel (DDP) environment.
    Handles both torchrun (env vars) and standard execution.
    """
    if torch.cuda.is_available():
        # 1. DDP Mode (via torchrun)
        if "LOCAL_RANK" in os.environ:
            dist.init_process_group(backend="nccl")
            local_rank = int(os.environ["LOCAL_RANK"])
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            torch.cuda.set_device(local_rank)
            return rank, world_size, torch.device(f"cuda:{local_rank}")
        
        # 2. Single GPU Mode (Standard python execution)
        else:
            # Default to first GPU
            torch.cuda.set_device(0) 
            return 0, 1, torch.device("cuda:0")
    
    # 3. CPU Fallback
    return 0, 1, torch.device("cpu")

def set_seed(seed: int):
    """
    Enforces deterministic behavior for reproducibility.
    Sets seeds for Python, NumPy, and PyTorch (CPU & GPU).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic algorithms (may impact performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def all_reduce_metrics(metrics: Dict[str, float], device: torch.device) -> Dict[str, float]:
    """
    Aggregates metrics across all DDP processes by averaging.
    Ensures that reported validation results cover the FULL dataset, not just the Rank-0 shard.
    """
    if not dist.is_initialized():
        return metrics

    # Sort keys to ensure same order across processes
    keys = sorted(metrics.keys())
    # Convert to tensor for broadcasting
    values = torch.tensor([metrics[k] for k in keys], device=device, dtype=torch.float32)
    
    # Sum across all GPUs
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    
    # Average
    values /= dist.get_world_size()
    
    return {k: v.item() for k, v in zip(keys, values)}

def load_pretrain_modules(path: str):
    """
    Dynamically loads class definitions from the pre-training script.
    Crucial for ensuring architecture consistency between pre-training 
    (reconstruction) and fine-tuning (downstream) phases.
    """
    spec = importlib.util.spec_from_file_location("pretrain_module", path)
    if spec is None:
        raise FileNotFoundError(f"Pre-train script not found at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    # Extract standard components expected by the factory
    return {
        'Config': mod.Config,
        'BertsWinMAE': BertsWinMAE,
        'MonaiMAEWrapper': getattr(mod, 'MonaiMAEWrapper', None),
        'MonaiViT': getattr(mod, 'MonaiViT', None),
        'patchify_image': getattr(mod, 'patchify_image', None),
        'TMJDataset': getattr(mod, 'TMJDataset', None),
        'process_batch_on_gpu': getattr(mod, 'process_batch_on_gpu', None),
        'collate_fn': getattr(mod, 'collate_fn', None),
        'hdf5_worker_init_fn': getattr(mod, 'hdf5_worker_init_fn', None),
    }, mod.Config()

# -----------------------------------------------------------------------------
# Optimizer Construction
# -----------------------------------------------------------------------------

def build_optimizer(model: torch.nn.Module, strategy: str, cfg: EvalConfig):
    """
    Constructs the optimizer with specific parameter groups based on the 
    fine-tuning strategy.
    
    Strategies:
    1. linear_probe: Freezes backbone, trains only the task-specific head.
    2. full: Trains everything. Head gets higher LR, backbone gets lower LR.
    3. last_stage: Unfreezes the head and the last N transformer blocks.
    4. lrd: Layer-wise Rate Decay (Differential LR for embeddings vs body vs head).
    5. lora: Freezes backbone (except injected adapters), trains head.
    """
    params = []
    
    # 1. Linear Probe (Head Only)
    if strategy == 'linear_probe':
        for param in model.backbone.parameters(): 
            param.requires_grad = False
        params.append({'params': model.head.parameters(), 'lr': cfg.LR_HEAD_ONLY})
        
    # 2. Full Finetuning
    elif strategy == 'full':
        for param in model.backbone.parameters(): 
            param.requires_grad = True
        params.append({'params': model.head.parameters(), 'lr': cfg.LR_HEAD_ONLY})
        params.append({'params': model.backbone.parameters(), 'lr': cfg.LR_FINETUNE})
        
    # 3. Last Stage Unfreezing
    elif strategy == 'last_stage':
        # First, freeze everything in backbone
        for param in model.backbone.parameters(): 
            param.requires_grad = False
        
        trainable_backbone_params = []
        
        # Logic to identify and unfreeze last blocks (Architecture dependent)
        # Handling MonaiViT / BertsWinMAE structures
        if hasattr(model.backbone, 'blocks'): # Standard ViT
            # Unfreeze last N blocks
            for block in model.backbone.blocks[-cfg.BLOCKS_TO_UNFREEZE:]:
                for p in block.parameters():
                    p.requires_grad = True
                    trainable_backbone_params.append(p)
            # Unfreeze final norm if present
            if hasattr(model.backbone, 'norm'):
                for p in model.backbone.norm.parameters():
                    p.requires_grad = True
                    trainable_backbone_params.append(p)
                    
        elif hasattr(model.backbone, 'features'): # BertsWinMAE/Swin
             # Assuming features is a sequential container of stages
             # We unfreeze the last stage
             last_stage = model.backbone.features[-1] if len(model.backbone.features) > 0 else None
             if last_stage:
                 for p in last_stage.parameters():
                     p.requires_grad = True
                     trainable_backbone_params.append(p)

        params.append({'params': model.head.parameters(), 'lr': cfg.LR_HEAD_ONLY})
        params.append({'params': trainable_backbone_params, 'lr': cfg.LR_FINETUNE})

    # 4. Layer-wise Rate Decay (LRD)
    elif strategy == 'lrd':
        for param in model.backbone.parameters(): 
            param.requires_grad = True
            
        # Differential Learning Rates
        lr_head = cfg.LR_HEAD_ONLY
        lr_body = (cfg.LR_FINETUNE + cfg.LR_HEAD_ONLY) / 10.0 # Mid-range
        lr_embed = cfg.LR_FINETUNE # Lowest
        
        head_params = list(model.head.parameters())
        embed_params = []
        body_params = []
        
        for name, p in model.backbone.named_parameters():
            if 'patch_embed' in name or 'cls_token' in name or 'pos_embed' in name:
                embed_params.append(p)
            else:
                body_params.append(p)
                
        params.append({'params': head_params, 'lr': lr_head})
        params.append({'params': body_params, 'lr': lr_body})
        params.append({'params': embed_params, 'lr': lr_embed})

    # 5. LoRA (Low-Rank Adaptation)
    elif strategy == 'lora':
        # Assumes LoRA layers are already injected or identified by name.
        # We freeze everything that is NOT a LoRA parameter.
        lora_params = []
        for name, p in model.backbone.named_parameters():
            if 'lora_' in name:
                p.requires_grad = True
                lora_params.append(p)
            else:
                p.requires_grad = False
                
        params.append({'params': model.head.parameters(), 'lr': cfg.LR_HEAD_ONLY})
        if lora_params:
            # Train LoRA params with the higher head LR usually
            params.append({'params': lora_params, 'lr': cfg.LR_HEAD_ONLY})
            
    else:
        raise ValueError(f"Unknown finetune strategy: {strategy}")
        
    return torch.optim.AdamW(params, weight_decay=cfg.WEIGHT_DECAY)

def log_config(config: EvalConfig):
    """
    Logs the effective configuration, showing default and overridden parameters.
    """
    logging.info("="*25 + " EFFECTIVE CONFIGURATION " + "="*25)
    
    # Get all class attributes (default config)
    default_config = {
        key: value for key, value in EvalConfig.__dict__.items()
        if not key.startswith('__') and not callable(value)
    }

    # Get instance-specific overrides
    overrides = config.__dict__
    
    # Combine and sort keys for consistent output
    all_keys = sorted(default_config.keys())
    max_key_length = max(len(key) for key in all_keys) if all_keys else 0

    for key in all_keys:
        value = getattr(config, key)
        marker = "[overridden]" if key in overrides else "[default]   "
        padding = " " * (max_key_length - len(key))
        logging.info(f"{key}:{padding} = {value} {marker}")
        
    logging.info("="*75)

# -----------------------------------------------------------------------------
# Main Orchestrator
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TMJ Downstream Evaluation Pipeline")
    parser.add_argument('--model', required=True, choices=EvalConfig.MODEL_REGISTRY.keys(), help="Key from MODEL_REGISTRY")
    parser.add_argument('--task', required=True, 
                        choices=[
                            'health_classification', 
                            'radiomics_baseline_regression', 
                            'radiomics_prognosis_regression',
                            'segmentation', 
                            'mri_inflam', 'mri_bone', 'mri_disc', 'mri_space'
                        ])
    parser.add_argument('--seg_head', default=None, choices=['linear', '2layer'], 
                        help="Override SEG_HEAD_TYPE for segmentation task (linear vs 2layer)")
    parser.add_argument('--strategy', default='linear_probe', choices=['linear_probe', 'full', 'last_stage', 'lrd', 'lora'])
    parser.add_argument('--aggregation', default='max', choices=['max', 'mean'], help="Siamese aggregation strategy (max/mean)")
    parser.add_argument('--lr_head', type=float, default=None, help="Override LR_HEAD_ONLY")
    parser.add_argument('--lr_fine', type=float, default=None, help="Override LR_FINETUNE")
    args = parser.parse_args()

    # 1. Initialization & Setup
    rank, world_size, device = setup_ddp()
    
    # Configure Config
    cfg = EvalConfig()
    if args.lr_head: cfg.LR_HEAD_ONLY = args.lr_head
    if args.lr_fine: cfg.LR_FINETUNE = args.lr_fine
    if args.seg_head:
        cfg.SEG_HEAD_TYPE = args.seg_head
        if rank == 0:
            logging.info(f"Ablation Override: SEG_HEAD_TYPE set to '{args.seg_head}'")

    # ACADEMIC NOTE: Radiomics Regression Consistency
    # Radiomic features (volume, surface area) are physically dependent on voxel size.
    # Applying geometric scaling during training/inference invalidates the regression targets (ground truth).
    # We strictly enforce 1.0 scaling (no scaling) for these tasks, regardless of the config file.
    if 'radiomics' in args.task or args.task == 'mri_space': 
        logging.info("Task is Regression: Enforcing fixed scale [1.0, 1.0].")
        cfg.AUG_SCALE_RANGE = [1.0, 1.0]
    
    # Logging Setup (Rank 0 only)
    run_id = f"{args.model}_{args.task}_{args.strategy}_{args.aggregation}"
    output_dir = cfg.BASE_OUTPUT_DIR / run_id
    
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(output_dir / "train.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info(f"Starting experiment: {run_id}")
        log_config(EvalConfig)
    else:
        logging.basicConfig(level=logging.ERROR)

    # Load Pre-training definitions
    pt_modules, pt_cfg = load_pretrain_modules(cfg.PRETRAIN_SCRIPT_PATH)
    
    # Hydrate Dataset Stats from Pretrain Config to ensure normalization consistency
    cfg.TARGET_IMAGE_SIZE = pt_cfg.INPUT_CUBE_SIZE
    cfg.HU_CLIP_MIN = pt_cfg.HU_CLIP_MIN
    cfg.HU_CLIP_MAX = pt_cfg.HU_CLIP_MAX
    cfg.DATASET_MEAN = pt_cfg.DATASET_MEAN
    cfg.DATASET_STD = pt_cfg.DATASET_STD
    cfg.EFFECTIVE_PATCH_SIZE = pt_cfg.EFFECTIVE_PATCH_SIZE

    TMJDataset_Class = pt_modules.get('TMJDataset')

    # 2. Data Preparation
    # Initialize Task Logic (manages labels, metrics, loss functions)
    if args.task == 'segmentation':
        task = TMJSegmentationTask(
            args.task, 
            cfg, 
            tmj_dataset_cls=TMJDataset_Class,
            gpu_process_fn=pt_modules['process_batch_on_gpu'],
            pt_config=pt_cfg
        )
        collate_func = pt_modules['collate_fn']
        worker_init = pt_modules['hdf5_worker_init_fn']
    elif args.task.startswith('mri_'):
        mri_task_registry = {
            'mri_inflam': TMJInflammationTask,
            'mri_bone': TMJBoneTask,
            'mri_disc': TMJDiscTask,
            'mri_space': TMJSpaceTask
        }
        
        if args.task not in mri_task_registry:
            raise ValueError(f"Implementation for {args.task} not found in registry.")

        task = mri_task_registry[args.task](
            config=cfg,
            tmj_dataset_cls=TMJDataset_Class,
            gpu_process_fn=pt_modules['process_batch_on_gpu'],
            pt_config=pt_cfg
        )
        collate_func = pt_modules['collate_fn']
        worker_init = pt_modules['hdf5_worker_init_fn']
    else:
        # --- Core Cohort (Classification/Regression) Data Loading ---
        task = TMJCoreTask(args.task, cfg)
        collate_func = None
        worker_init = worker_init_fn
    
    manifest, stratify_labels = task.load_manifest()
    
    # 3. Repeated K-Fold Cross-Validation Loop
    # We use CV_REPEATS to ensure statistical robustness of the results
    all_repeats_metrics = []
    
    for repeat in range(cfg.CV_REPEATS):

        repeat_result_file = output_dir / f"repeat_{repeat}_results.json"
        if repeat_result_file.exists():
            if rank == 0:
                logging.info(f"Found existing results for Repeat {repeat+1}/{cfg.CV_REPEATS}. Loading and skipping...")
                try:
                    with open(repeat_result_file, 'r') as f:
                        prev_results = json.load(f)
                    all_repeats_metrics.append(prev_results)
                except json.JSONDecodeError:
                    logging.error(f"Corrupted JSON for Repeat {repeat+1}. Please delete {repeat_result_file} and restart.")
                    sys.exit(1)
            
            if world_size > 1: 
                dist.barrier()
            
            continue

        if rank == 0: logging.info(f"=== Starting Repeat {repeat+1}/{cfg.CV_REPEATS} ===")
        
        # Seed varies per repeat to ensure different fold splits
        repeat_seed = cfg.SEED + repeat
        set_seed(repeat_seed)

        folds_generator = task.get_fold_iterator(
            manifest, 
            stratify_labels, 
            num_folds=cfg.K_FOLDS, 
            seed=repeat_seed
        )


        repeat_metrics_accum = defaultdict(list)
        
        for fold, (train_idx, val_idx) in enumerate(folds_generator):
            if rank == 0: logging.info(f"--- Fold {fold+1} ---")
            train_ds_fold, val_ds_fold = task.get_datasets(train_idx, val_idx, manifest)
            scaler_stats = None

            if task.type == 'regression':
                train_targets = np.stack([manifest[i][task.target_key].numpy() for i in train_idx])
                scaler_stats = {
                    'mean': torch.tensor(np.mean(train_targets, axis=0), device=device, dtype=torch.float32),
                    'std': torch.tensor(np.std(train_targets, axis=0) + 1e-6, device=device, dtype=torch.float32)
                }

            # Standard DDP Samplers
            train_sampler = DistributedSampler(train_ds_fold, num_replicas=world_size, rank=rank, shuffle=True)
            val_sampler = DistributedSampler(val_ds_fold, num_replicas=world_size, rank=rank, shuffle=False)
            
            train_loader = DataLoader(
                train_ds_fold, 
                batch_size=cfg.GPU_BATCH_SIZE, 
                sampler=train_sampler, 
                num_workers=cfg.NUM_WORKERS, 
                pin_memory=True, 
                drop_last=True,
                worker_init_fn=worker_init,
                collate_fn=collate_func
            )
            val_loader = DataLoader(
                val_ds_fold, 
                batch_size=cfg.GPU_BATCH_SIZE, 
                sampler=val_sampler, 
                num_workers=cfg.NUM_WORKERS, 
                pin_memory=True,
                worker_init_fn=worker_init,
                collate_fn=collate_func
            )

            # C. Model Initialization
            # Re-initialize model for every fold to ensure no weight leakage
            m_info = EvalConfig.MODEL_REGISTRY[args.model]
            
            # Determine Inference Mode
            # For segmentation, we MUST override the registry default to enable spatial output.
            inference_mode = 'segmentation' if args.task == 'segmentation' else m_info.get('inference_mode', 'full_image')

            backbone, feature_dim = ModelFactory.load_backbone(
                m_info['arch_type'], m_info['path'], pt_cfg, pt_modules
            )
            
            if args.strategy == 'lora':
                if rank == 0: logging.info(f"Injecting LoRA modules into backbone (rank=8, alpha=16)...")
                inject_lora_to_model(backbone, rank=8, alpha=16)
            
            patchify_fn = pt_modules.get('patchify_image', None)
            # Head is created dynamically based on task output dimension
            head = task.get_head(feature_dim)
            model = TMJAnalysisModel(
                backbone, 
                head, 
                feature_dim, 
                m_info['arch_type'],
                inference_mode=inference_mode,
                patchify_fn=patchify_fn,
                patch_size=cfg.EFFECTIVE_PATCH_SIZE
            ).to(device)
            
            # Wrap DDP
            if world_size > 1:
                model = DDP(model, device_ids=[int(os.environ.get("LOCAL_RANK", 0))], find_unused_parameters=True)
                
            # D. Optimization Setup
            optimizer = build_optimizer(model.module if world_size > 1 else model, args.strategy, cfg)
            scaler = torch.amp.GradScaler('cuda')
            # ==================================================================
            # ACADEMIC NOTE: EARLY STOPPING CRITERIA
            # ==================================================================
            # Preliminary experiments showed that monitoring the target metric (Validation AUC)
            # exhibited high variance due to the small validation fold size (N~14).
            # This noise led to premature convergence on suboptimal epochs.
            # Validation Loss proved to be a more robust indicator of overfitting for this dataset.
            # Therefore, we strictly use `mode='min'` (Val Loss) for all tasks.
            # ==================================================================
            early_stop = EarlyStopping(patience=cfg.PATIENCE_EPOCHS, mode='min') # Always min (loss)

            # Define Checkpoint Path for this Fold
            fold_ckpt_path = output_dir / f"checkpoint_rep{repeat}_fold{fold}.pth"
            
            # E. Training Loop
            best_metric = None
            
            for epoch in range(cfg.NUM_EPOCHS):
                train_sampler.set_epoch(epoch)
                
                train_loss = train_epoch(
                    model, 
                    train_loader, 
                    optimizer, 
                    scaler, 
                    task, 
                    device, 
                    aggregation=args.aggregation,
                    scaler_stats=scaler_stats
                )
                
                # Validate (supports TTA if configured)
                val_results = validate_epoch(
                    model, 
                    val_loader, 
                    task, 
                    device, 
                    aggregation=args.aggregation,
                    scaler_stats=scaler_stats, 
                    tta_runs=cfg.TTA_RUNS,
                    compute_surface_metrics=False
                )
                
                val_loss = val_results['loss']
                if task.type == 'classification':
                    primary_metric = val_results.get('auc', 0.0)
                elif task.type == 'regression':
                    primary_metric = val_results.get('r2', 0.0)
                elif task.type == 'segmentation':
                    primary_metric = val_results.get('dsc_mean', 0.0) # Use Mean Dice
                else:
                    raise("No metrics")

                if rank == 0:
                    logging.info(
                        f"Epoch {epoch+1}: Train Loss {train_loss:.4f} | "
                        f"Val Loss {val_loss:.4f} | Metric {primary_metric:.4f}"
                    )
                
                # Early Stopping Check
                # Validation Loss used for Early Stopping as AUC/R2 can be noisy on small batch sizes
                if early_stop(val_loss):
                    best_metric = val_results
                    if rank == 0:
                        state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': state_dict,
                            'config': vars(cfg),
                            'best_metric': val_results
                        }, fold_ckpt_path)
                        logging.info(f"Saved best model to {fold_ckpt_path}")
                    
                if early_stop.early_stop:
                    if rank == 0: logging.info("Early stopping triggered.")
                    break
            if best_metric is None:
                best_metric = val_results

            if world_size > 1: dist.barrier()


            if rank == 0:
                logging.info("Reloading best model for Surface Metric (HD95/ASSD) calculation...")

            if fold_ckpt_path.exists():
                ckpt = torch.load(fold_ckpt_path, map_location=device, weights_only=False)
                if world_size > 1:
                    model.module.load_state_dict(ckpt['model_state_dict'])
                else:
                    model.load_state_dict(ckpt['model_state_dict'])
            else:
                logging.warning(f"Checkpoint not found at {fold_ckpt_path}. Using last model state.")
            
            is_seg_task = (args.task == 'segmentation')
            local_metrics = validate_epoch(
                model, val_loader, task, device,
                aggregation=args.aggregation,
                scaler_stats=None,
                tta_runs=1,
                compute_surface_metrics=is_seg_task 
            )
            # 3. Reduce Metrics across GPUs
            # This averages the Dice/HD95 from all shards to get the true dataset mean
            final_fold_metrics = all_reduce_metrics(local_metrics, device)

            # End of Fold
            if rank == 0:
                logging.info(f"Fold {fold+1} Final Aggregated Metrics: {final_fold_metrics}")
                for k, v in final_fold_metrics.items():
                    repeat_metrics_accum[k].append(v)
            
            del model
            torch.cuda.empty_cache()

            # Sync processes before next fold
            if world_size > 1: dist.barrier()
            
        # End of Repeat - Aggregate metrics for this repeat
        if rank == 0:
            repeat_summary = {k: np.mean(v) for k, v in repeat_metrics_accum.items()}
            all_repeats_metrics.append(repeat_summary)
            with open(output_dir / f"repeat_{repeat}_results.json", "w") as f:
                json.dump(repeat_summary, f, indent=4)
            logging.info(f"Repeat {repeat+1} Summary: {repeat_summary}")

    # 4. Final Results Saving
    if rank == 0:
        # Calculate Mean/Std across repeats
        final_stats = {}
        if all_repeats_metrics:
            metric_keys = all_repeats_metrics[0].keys()
            for k in metric_keys:
                values = [m[k] for m in all_repeats_metrics]
                final_stats[f"{k}_mean"] = float(np.mean(values))
                final_stats[f"{k}_std"] = float(np.std(values))
                ci95 = 1.96 * (np.std(values) / np.sqrt(len(values)))
                final_stats[f"{k}_ci95"] = float(ci95)
            
        logging.info("=== Final Aggregated Results ===")
        logging.info(json.dumps(final_stats, indent=4))
        
        # Save to disk
        results_payload = {
            'config': vars(cfg),
            'args': vars(args),
            'final_stats': final_stats,
            'raw_repeats': all_repeats_metrics
        }
        
        with open(output_dir / "final_results.json", "w") as f:
            json.dump(results_payload, f, indent=4)

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    # Ensure NCCL settings for robust DDP
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    main()
    