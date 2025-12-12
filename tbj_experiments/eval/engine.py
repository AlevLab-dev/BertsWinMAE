import torch
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, Optional, Any, List
from torch.amp import autocast

def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    task: Any,
    device: torch.device,
    aggregation: str = 'max',
    scaler_stats: Optional[Dict[str, torch.Tensor]] = None
) -> float:
    """
    Executes a single training epoch using mixed precision and regression target scaling.

    ACADEMIC NOTE:
    For regression tasks (Radiomics), target variables are normalized using Z-score standardization
    based on training set statistics to ensure gradient stability. The loss is computed in this
    normalized latent space.

    Args:
        model: The TMJAnalysisModel neural network.
        loader: Training DataLoader.
        optimizer: The optimizer instance (e.g., AdamW).
        scaler: GradScaler for mixed-precision training.
        task: The Task instance handling batch processing and loss definitions.
        device: Computational device (GPU).
        aggregation: Strategy for bilateral feature fusion ('max' or 'mean').
        scaler_stats: Dictionary containing 'mean' and 'std' tensors for regression normalization.
                      Should be None for classification tasks.

    Returns:
        avg_loss: The average loss value over the epoch.
    """
    model.train()
    loss_fn = task.get_loss_fn(device)
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(loader):
        # 1. Task-specific Preprocessing (Augmentation ON for Training)
        # The task delegates to PhysicsEngine to handle anatomical mirroring and affine transforms.
        data_dict = task.process_batch(batch, device, augment=True)
        
        x = data_dict['x']
        x_pair = data_dict.get('x_pair')
        mask = data_dict.get('mask')
        mask_pair = data_dict.get('mask_pair')
        labels_raw = data_dict['labels']
        
        # 2. Target Normalization (Regression Only)
        # Transforms physical units (e.g., HU, mm) into Z-scores: z = (x - mean) / std.
        if scaler_stats is not None:
            labels_for_loss = (labels_raw - scaler_stats['mean']) / scaler_stats['std']
        else:
            labels_for_loss = labels_raw
            
        # 3. Forward & Backward Pass
        optimizer.zero_grad(set_to_none=True)
        
        # We use bfloat16 for training stability as verified in pre-training experiments.
        with autocast('cuda', dtype=torch.bfloat16):
            preds = model(x, x_pair, mask=mask, mask_pair=mask_pair, aggregation=aggregation)
            loss = loss_fn(preds, labels_for_loss)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    task: Any,
    device: torch.device,
    aggregation: str = 'max',
    scaler_stats: Optional[Dict[str, torch.Tensor]] = None,
    tta_runs: int = 1,
    compute_surface_metrics: bool = False
) -> Dict[str, float]:
    """
    Executes a validation epoch with support for Test-Time Augmentation (TTA).

    METHODOLOGY:
    1. Loss Calculation: Performed on the canonical (non-augmented) input against 
       normalized targets (if regression) to ensure consistency with the training objective 
       and stable Early Stopping.
    2. Metric Calculation: Performed on de-normalized predictions (restored to original units)
       against raw ground truth labels.
    3. TTA: If enabled, averages predictions across multiple augmented views to reduce 
       variance and improve robustness.

    Args:
        model: The TMJAnalysisModel model.
        loader: Validation DataLoader.
        task: The Task instance.
        device: Computational device.
        aggregation: Strategy for bilateral feature fusion.
        scaler_stats: Mean/Std statistics for regression de-normalization.
        tta_runs: Number of forward passes (1 = Canonical only; >1 = Canonical + N-1 Augmented).

    Returns:
        metrics: Dictionary containing calculated metrics (e.g., AUC, R2, MAE) and Loss.
    """
    model.eval()
    loss_fn = task.get_loss_fn(device)

    # Storage for scalar tasks (Classif/Reg)
    all_preds_raw = []
    all_labels_raw = []
    # Storage for Segmentation accumulation
    seg_metrics_accum = defaultdict(list)

    total_loss = 0.0
    
    for batch in loader:
        # 1. Prepare Ground Truth
        # Task-specific handling: Segmentation uses dense masks, others use target keys
        if task.type == 'segmentation':
            # process_batch handles the transfer/prep, so we just grab raw for now if needed,
            # but strictly we rely on process_batch output for consistency.
            pass
        else:
            labels_original = batch[task.target_key].to(device)
            if scaler_stats is not None:
                labels_for_loss = (labels_original - scaler_stats['mean']) / scaler_stats['std']
            else:
                labels_for_loss = labels_original

        # Pass 1: Canonical (Augmentation OFF)
        # Represents the exact anatomical inputs without geometric distortion.
        data_dict = task.process_batch(batch, device, augment=False)

        x = data_dict['x']
        x_pair = data_dict.get('x_pair')
        mask = data_dict.get('mask')
        mask_pair = data_dict.get('mask_pair')
        labels_processed = data_dict['labels']
        
        with autocast('cuda', dtype=torch.bfloat16):
            pred_canonical = model(x, x_pair, mask=mask, mask_pair=mask_pair, aggregation=aggregation)
            if scaler_stats is not None:
                # Only relevant for regression logic where we computed labels_for_loss above
                loss = loss_fn(pred_canonical, labels_for_loss)
            else:
                loss = loss_fn(pred_canonical, labels_processed)
        
        total_loss += loss.item()
        
        # 3. Metric Calculation Logic
        # BRANCH A: Segmentation (Batch-wise calculation to prevent OOM)
        if task.type == 'segmentation':
            # TTA is typically skipped or handled differently for segmentation efficiency
            # Here we calculate metrics immediately on the GPU/CPU batch
            batch_metrics = task.calculate_metrics(
                labels_processed.cpu().numpy(), 
                pred_canonical.float().cpu().numpy()
            )
            for k, v in batch_metrics.items():
                seg_metrics_accum[k].append(v)
            
            # B. Advanced Surface Metrics (HD95/ASSD) - Only if requested
            # Computed batch-wise to keep memory usage low
            if compute_surface_metrics:
                geom_metrics = task.calculate_geometric_metrics(
                    labels_processed.cpu().numpy(), 
                    pred_canonical.float().cpu().numpy()
                )
                for k, v in geom_metrics.items():
                    seg_metrics_accum[k].append(v)
                
        # BRANCH B: Classification / Regression (Accumulate all)
        else:
            # TTA Logic (Only applied here for scalar tasks as per original design)
            final_preds = pred_canonical
            
            if tta_runs > 1:
                preds_list = [pred_canonical]
                for _ in range(tta_runs - 1):
                    dd_aug = task.process_batch(batch, device, augment=True)
                    with autocast('cuda', dtype=torch.bfloat16):
                        p_aug = model(
                            dd_aug['x'], dd_aug.get('x_pair'), 
                            mask=dd_aug.get('mask'), mask_pair=dd_aug.get('mask_pair'), 
                            aggregation=aggregation
                        )
                        preds_list.append(p_aug)
                final_preds = torch.stack(preds_list).mean(dim=0)

            # De-normalization
            if scaler_stats is not None:
                final_preds = (final_preds * scaler_stats['std']) + scaler_stats['mean']
            
            all_preds_raw.append(final_preds.float().cpu())
            all_labels_raw.append(labels_original.float().cpu())
            
    # 4. Aggregate Final Metrics
    metrics = {'loss': total_loss / len(loader)}
    
    if task.type == 'segmentation':
        # Average the batch-wise metrics
        for k, v in seg_metrics_accum.items():
            metrics[k] = np.nanmean(v)
    else:
        # Concatenate and compute scalar metrics
        if all_preds_raw:
            preds_cat = torch.cat(all_preds_raw).numpy()
            labels_cat = torch.cat(all_labels_raw).numpy()
            scalar_metrics = task.calculate_metrics(labels_cat, preds_cat)
            metrics.update(scalar_metrics)
            
    return metrics


class EarlyStopping:
    """
    Implements Early Stopping to mitigate overfitting.
    
    The mechanism monitors a specified metric (typically validation loss) and stops training 
    if no improvement is observed within the 'patience' window. 
    Supports both minimization (e.g., Loss, MAE) and maximization (e.g., AUC, R2) objectives.
    """
    def __init__(self, patience: int = 7, mode: str = 'min', delta: float = 0.0):
        """
        Args:
            patience (int): Epochs to wait after the last improvement.
            mode (str): Optimization direction. 'min' for Loss/Error, 'max' for Scores.
            delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        
    def __call__(self, current_value: float) -> bool:
        """
        Updates the internal state based on the current epoch's metric.
        
        Args:
            current_value (float): The metric value to evaluate.
            
        Returns:
            is_best (bool): True if the current model represents the best state so far.
        """
        # Robustness check for numerical stability
        if np.isnan(current_value):
            return False

        # Normalize score direction so that higher is always better for comparison logic
        score = -current_value if self.mode == 'min' else current_value

        if self.best_score is None:
            self.best_score = score
            self.best_value = current_value
            return True

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.best_value = current_value
            self.counter = 0
            return True