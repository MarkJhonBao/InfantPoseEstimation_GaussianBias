"""
Validation Script for Human Pose Estimation
"""

import os
import sys
import argparse
import logging

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config
from datasets import build_dataloader
from models import build_model
from utils import COCOEvaluator, AverageMeter


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    return logging.getLogger(__name__)


def transform_preds(coords, center, scale, output_size):
    """Transform predictions back to original image coordinates."""
    target_coords = coords.copy()
    target_coords[0] = coords[0] / output_size[0] * scale[0] + center[0] - scale[0] / 2
    target_coords[1] = coords[1] / output_size[1] * scale[1] + center[1] - scale[1] / 2
    return target_coords


def validate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg,
    logger: logging.Logger,
    flip_test: bool = True,
):
    """Validate the model with detailed metrics."""
    model.eval()
    
    loss_meter = AverageMeter('Loss', ':.4f')
    
    # COCO evaluator
    evaluator = COCOEvaluator(
        ann_file=os.path.join(cfg.data.data_root, cfg.data.val_ann),
        num_keypoints=cfg.data.num_keypoints,
    )
    
    logger.info(f'Validating on {len(dataloader.dataset)} samples...')
    logger.info(f'Flip test: {flip_test}')
    
    flip_pairs = cfg.data.flip_pairs if flip_test else None
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            imgs = batch['img'].to(device)
            targets = batch['target'].to(device)
            target_weights = batch['target_weight'].to(device)
            metas = batch['meta']
            
            # Get keypoints for fusion loss
            gt_keypoints = batch.get('keypoints')
            if gt_keypoints is not None:
                gt_keypoints = gt_keypoints.to(device)
            
            # Forward pass with optional flip test
            if flip_test:
                pred_keypoints, pred_scores = model.inference(
                    imgs, flip=True, flip_pairs=flip_pairs
                )
            else:
                outputs = model(imgs)
                # Check if model has fusion head
                if hasattr(model, 'head') and hasattr(model.head, 'decode'):
                    pred_keypoints, pred_scores = model.head.decode(outputs, apply_offset=True)
                else:
                    heatmaps = outputs['heatmaps']
                    pred_keypoints, pred_scores = model.decode_heatmaps(heatmaps)
            
            # Compute loss (for logging)
            outputs = model(
                imgs, targets, target_weights,
                gt_keypoints=gt_keypoints,
                input_size=cfg.data.input_size,
            )
            loss = outputs['loss']
            loss_meter.update(loss.item(), imgs.size(0))
            
            # Convert to numpy
            pred_keypoints = pred_keypoints.cpu().numpy()
            pred_scores = pred_scores.cpu().numpy()
            
            # Scale to input size
            scale_x = cfg.data.input_size[0] / cfg.data.heatmap_size[0]
            scale_y = cfg.data.input_size[1] / cfg.data.heatmap_size[1]
            pred_keypoints[:, :, 0] *= scale_x
            pred_keypoints[:, :, 1] *= scale_y
            
            # Transform back to original image coordinates
            batch_size = imgs.size(0)
            for i in range(batch_size):
                center = metas['center'][i].numpy()
                scale = metas['scale'][i].numpy()
                
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
            
            # Progress
            if batch_idx % 50 == 0:
                logger.info(f'  [{batch_idx}/{len(dataloader)}] Loss: {loss_meter.avg:.4f}')
    
    # Compute metrics
    metrics = evaluator.evaluate()
    
    return metrics, loss_meter.avg


def main(args):
    """Main validation function."""
    logger = setup_logging()
    
    # Load config
    cfg = get_config()
    
    if args.data_root:
        cfg.data.data_root = args.data_root
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Build dataloader
    logger.info('Building validation dataloader...')
    val_loader = build_dataloader(cfg, is_train=False)
    logger.info(f'Validation samples: {len(val_loader.dataset)}')
    
    # Build model
    logger.info('Building model...')
    model = build_model(cfg)
    model = model.to(device)
    
    # Load checkpoint
    if args.checkpoint:
        logger.info(f'Loading checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'Loaded checkpoint from epoch {checkpoint.get("epoch", "unknown")}')
    else:
        logger.warning('No checkpoint provided, using randomly initialized model')
    
    # Validate
    metrics, val_loss = validate(
        model, val_loader, device, cfg, logger,
        flip_test=not args.no_flip
    )
    
    # Print results
    logger.info('\n' + '=' * 60)
    logger.info('VALIDATION RESULTS')
    logger.info('=' * 60)
    logger.info(f'Loss: {val_loss:.4f}')
    logger.info(f'AP:   {metrics["AP"]:.4f}')
    logger.info(f'AP50: {metrics["AP50"]:.4f}')
    logger.info(f'AP75: {metrics["AP75"]:.4f}')
    
    if 'AP_M' in metrics:
        logger.info(f'AP_M: {metrics["AP_M"]:.4f}')
        logger.info(f'AP_L: {metrics["AP_L"]:.4f}')
    
    if 'AR' in metrics:
        logger.info(f'AR:   {metrics["AR"]:.4f}')
        logger.info(f'AR50: {metrics["AR50"]:.4f}')
        logger.info(f'AR75: {metrics["AR75"]:.4f}')
    
    logger.info('=' * 60)
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate Pose Estimation Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--data_root', type=str, default=None, help='Data root directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--no_flip', action='store_true', help='Disable flip test')
    
    args = parser.parse_args()
    main(args)
