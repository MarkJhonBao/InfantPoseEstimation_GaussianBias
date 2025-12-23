"""
Training Script for Human Pose Estimation
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config, Config
from datasets import build_dataloader
from models import build_model
from utils import AverageMeter, MetricLogger


def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, 'train.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, cfg: Config) -> optim.Optimizer:
    """Build optimizer."""
    train_cfg = cfg.train
    
    # Separate parameters with different weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'bn' in name or 'norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': train_cfg.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    
    if train_cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            param_groups,
            lr=train_cfg.lr,
            betas=train_cfg.betas,
        )
    elif train_cfg.optimizer == 'Adam':
        optimizer = optim.Adam(
            param_groups,
            lr=train_cfg.lr,
            betas=train_cfg.betas,
        )
    elif train_cfg.optimizer == 'SGD':
        optimizer = optim.SGD(
            param_groups,
            lr=train_cfg.lr,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: {train_cfg.optimizer}")
    
    return optimizer


def build_scheduler(
    optimizer: optim.Optimizer,
    cfg: Config,
    num_iters_per_epoch: int,
) -> optim.lr_scheduler._LRScheduler:
    """Build learning rate scheduler."""
    train_cfg = cfg.train
    
    # Warmup + MultiStep scheduler
    warmup_iters = train_cfg.warmup_epochs * num_iters_per_epoch
    total_iters = train_cfg.max_epochs * num_iters_per_epoch
    milestones = [m * num_iters_per_epoch for m in train_cfg.lr_milestones]
    
    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            # Linear warmup
            return train_cfg.warmup_lr / train_cfg.lr + \
                   (1 - train_cfg.warmup_lr / train_cfg.lr) * current_iter / warmup_iters
        else:
            # Multi-step decay
            factor = 1.0
            for milestone in milestones:
                if current_iter >= milestone:
                    factor *= train_cfg.lr_gamma
            return factor
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return scheduler


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    cfg: Config,
    logger: logging.Logger,
    writer: SummaryWriter,
):
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter('Loss', ':.4f')
    batch_time = AverageMeter('Time', ':.3f')
    data_time = AverageMeter('Data', ':.3f')
    
    num_batches = len(dataloader)
    log_interval = max(1, num_batches // 10)
    
    end = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        data_time.update(time.time() - end)
        
        # Move data to device
        imgs = batch['img'].to(device)
        targets = batch['target'].to(device)
        target_weights = batch['target_weight'].to(device)
        
        # Forward pass with mixed precision
        optimizer.zero_grad()
        
        with autocast(enabled=cfg.train.fp16):
            outputs = model(imgs, targets, target_weights)
            loss = outputs['loss']
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update scheduler
        scheduler.step()
        
        # Update meters
        loss_meter.update(loss.item(), imgs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Log progress
        if batch_idx % log_interval == 0 or batch_idx == num_batches - 1:
            lr = optimizer.param_groups[0]['lr']
            global_step = epoch * num_batches + batch_idx
            
            logger.info(
                f'Epoch [{epoch}][{batch_idx}/{num_batches}] '
                f'Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                f'LR: {lr:.6f} '
                f'Time: {batch_time.val:.3f}s'
            )
            
            # TensorBoard logging
            writer.add_scalar('train/loss', loss_meter.val, global_step)
            writer.add_scalar('train/lr', lr, global_step)
    
    return loss_meter.avg


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: Config,
    logger: logging.Logger,
):
    """Validate the model."""
    model.eval()
    
    loss_meter = AverageMeter('Loss', ':.4f')
    
    # For COCO evaluation
    from utils import COCOEvaluator
    evaluator = COCOEvaluator(
        ann_file=os.path.join(cfg.data.data_root, cfg.data.val_ann),
        num_keypoints=cfg.data.num_keypoints,
    )
    
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch['img'].to(device)
            targets = batch['target'].to(device)
            target_weights = batch['target_weight'].to(device)
            metas = batch['meta']
            
            # Forward pass
            outputs = model(imgs, targets, target_weights)
            loss = outputs['loss']
            heatmaps = outputs['heatmaps']
            
            loss_meter.update(loss.item(), imgs.size(0))
            
            # Decode predictions
            pred_keypoints, pred_scores = model.decode_heatmaps(heatmaps)
            
            # Transform to original image coordinates
            pred_keypoints = pred_keypoints.cpu().numpy()
            pred_scores = pred_scores.cpu().numpy()
            
            # Scale to input size, then to original image
            scale_x = cfg.data.input_size[0] / cfg.data.heatmap_size[0]
            scale_y = cfg.data.input_size[1] / cfg.data.heatmap_size[1]
            pred_keypoints[:, :, 0] *= scale_x
            pred_keypoints[:, :, 1] *= scale_y
            
            # Transform back to original image coordinates
            batch_size = imgs.size(0)
            for i in range(batch_size):
                center = metas['center'][i].numpy()
                scale = metas['scale'][i].numpy()
                
                # Inverse affine transform
                for k in range(cfg.data.num_keypoints):
                    pred_keypoints[i, k] = transform_preds(
                        pred_keypoints[i, k],
                        center, scale,
                        cfg.data.input_size
                    )
            
            # Update evaluator
            evaluator.update(
                pred_keypoints=pred_keypoints,
                pred_scores=pred_scores,
                image_ids=[m.item() for m in metas['image_id']],
                ann_ids=[m.item() for m in metas['ann_id']],
                centers=np.stack([m.numpy() for m in metas['center']]),
                scales=np.stack([m.numpy() for m in metas['scale']]),
                areas=np.array([m.item() for m in metas['area']]),
                bboxes=np.stack([m.numpy() for m in metas['bbox']]),
            )
    
    # Compute metrics
    metrics = evaluator.evaluate()
    
    logger.info(f'Validation Loss: {loss_meter.avg:.4f}')
    logger.info(f'Validation AP: {metrics["AP"]:.4f}')
    logger.info(f'Validation AP50: {metrics["AP50"]:.4f}')
    logger.info(f'Validation AP75: {metrics["AP75"]:.4f}')
    
    return metrics


def transform_preds(coords, center, scale, output_size):
    """Transform predictions back to original image coordinates."""
    target_coords = coords.copy()
    
    # Scale from output_size to scale
    target_coords[0] = coords[0] / output_size[0] * scale[0] + center[0] - scale[0] / 2
    target_coords[1] = coords[1] / output_size[1] * scale[1] + center[1] - scale[1] / 2
    
    return target_coords


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: dict,
    output_dir: str,
    is_best: bool = False,
):
    """Save checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
    }
    
    # Save latest
    torch.save(checkpoint, os.path.join(output_dir, 'latest.pth'))
    
    # Save best
    if is_best:
        torch.save(checkpoint, os.path.join(output_dir, 'best.pth'))
    
    # Save epoch checkpoint
    if epoch % 10 == 0:
        torch.save(checkpoint, os.path.join(output_dir, f'epoch_{epoch}.pth'))


def main(args):
    """Main training function."""
    # Load config
    cfg = get_config()
    
    # Override config with args
    if args.data_root:
        cfg.data.data_root = args.data_root
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.epochs:
        cfg.train.max_epochs = args.epochs
    if args.lr:
        cfg.train.lr = args.lr
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(cfg.train.checkpoint_dir, f'{cfg.exp_name}_{timestamp}')
    
    logger = setup_logging(output_dir)
    logger.info(f'Config: {cfg}')
    
    set_seed(cfg.seed)
    
    device = torch.device(cfg.train.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    
    # Build dataloader
    logger.info('Building dataloaders...')
    train_loader = build_dataloader(cfg, is_train=True)
    val_loader = build_dataloader(cfg, is_train=False)
    logger.info(f'Train samples: {len(train_loader.dataset)}')
    logger.info(f'Val samples: {len(val_loader.dataset)}')
    
    # Build model
    logger.info('Building model...')
    model = build_model(cfg)
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Number of trainable parameters: {num_params:,}')
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = GradScaler(enabled=cfg.train.fp16)
    
    # Resume from checkpoint
    start_epoch = 0
    best_ap = 0.0
    
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f'Loading checkpoint: {args.resume}')
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_ap = checkpoint['metrics'].get('AP', 0.0)
            logger.info(f'Resumed from epoch {start_epoch}')
    
    # Training loop
    logger.info('Starting training...')
    
    for epoch in range(start_epoch, cfg.train.max_epochs):
        logger.info(f'\n{"="*50}')
        logger.info(f'Epoch {epoch}/{cfg.train.max_epochs}')
        logger.info(f'{"="*50}')
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch, cfg, logger, writer
        )
        
        # Validate
        if (epoch + 1) % cfg.train.val_interval == 0 or epoch == cfg.train.max_epochs - 1:
            metrics = validate(model, val_loader, device, cfg, logger)
            
            # TensorBoard
            writer.add_scalar('val/loss', metrics.get('loss', 0), epoch)
            writer.add_scalar('val/AP', metrics['AP'], epoch)
            writer.add_scalar('val/AP50', metrics['AP50'], epoch)
            writer.add_scalar('val/AP75', metrics['AP75'], epoch)
            
            # Save checkpoint
            is_best = metrics['AP'] > best_ap
            if is_best:
                best_ap = metrics['AP']
                logger.info(f'New best AP: {best_ap:.4f}')
            
            save_checkpoint(
                model, optimizer, scheduler, epoch, metrics,
                output_dir, is_best
            )
    
    logger.info(f'\nTraining completed! Best AP: {best_ap:.4f}')
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Pose Estimation Model')
    parser.add_argument('--data_root', type=str, default=None, help='Data root directory')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    main(args)
