import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import json

from models.pose_hrnet import PoseHighResolutionNet
from models.losses import FusedPoseLoss, MorphologyShapeLoss
from data.coco_dataset import PreemieCocoDataset
from utils.metrics import calculate_pck, calculate_ap
from utils.postprocess import fused_decode
from config import get_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train Preterm Infant Pose Estimation')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='configuration file path')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='directory containing COCO format annotations')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='directory to save checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='checkpoint path to resume training')
    parser.add_argument('--gpus', type=str, default='0',
                        help='GPU ids to use')
    return parser.parse_args()


class Trainer:
    def __init__(self, config, model, train_loader, val_loader, device):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Initialize losses
        self.heatmap_loss = FusedPoseLoss(
            use_target_weight=True,
            loss_type='mse'
        )
        self.morph_loss = MorphologyShapeLoss(
            lambda_variance=config.LOSS.MORPH_LAMBDA
        )
        self.regression_loss = nn.SmoothL1Loss()
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.TRAIN.EPOCHS
        )
        
        self.start_epoch = 0
        self.best_ap = 0.0
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            target_heatmaps = batch['target_heatmap'].to(self.device)
            target_coords = batch['target_coords'].to(self.device)
            target_weight = batch['target_weight'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            pred_heatmaps = outputs['heatmaps']
            pred_coords = outputs.get('coords', None)
            
            # Calculate losses
            loss_heatmap = self.heatmap_loss(pred_heatmaps, target_heatmaps, target_weight)
            loss_morph = self.morph_loss(pred_heatmaps, target_heatmaps, target_weight)
            
            loss = loss_heatmap + self.config.LOSS.MORPH_WEIGHT * loss_morph
            
            # Add regression loss if model outputs coordinates
            if pred_coords is not None:
                loss_reg = self.regression_loss(pred_coords, target_coords)
                loss += self.config.LOSS.REG_WEIGHT * loss_reg
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self, epoch):
        self.model.eval()
        all_preds = []
        all_boxes = []
        all_image_ids = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                
                outputs = self.model(images)
                
                # Fused decoding: combine heatmap and regression
                preds, maxvals = fused_decode(
                    outputs['heatmaps'],
                    outputs.get('coords', None),
                    batch['center'],
                    batch['scale']
                )
                
                all_preds.append(preds.cpu().numpy())
                all_boxes.append(batch['bbox'].cpu().numpy())
                all_image_ids.append(batch['image_id'].cpu().numpy())
        
        # Calculate metrics
        ap = calculate_ap(all_preds, all_boxes, all_image_ids, self.val_loader.dataset)
        pck = calculate_pck(all_preds, all_boxes, all_image_ids, self.val_loader.dataset)
        
        print(f'Epoch {epoch} - AP: {ap:.4f}, PCK@0.2: {pck:.4f}')
        return ap, pck
    
    def save_checkpoint(self, epoch, ap, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_ap': self.best_ap,
            'config': self.config
        }
        
        save_path = os.path.join(self.config.OUTPUT_DIR, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = os.path.join(self.config.OUTPUT_DIR, 'model_best.pth')
            torch.save(checkpoint, best_path)
            print(f'Best model saved with AP: {ap:.4f}')
    
    def train(self):
        for epoch in range(self.start_epoch, self.config.TRAIN.EPOCHS):
            # Train
            train_loss = self.train_epoch(epoch)
            print(f'Epoch {epoch} - Training Loss: {train_loss:.4f}')
            
            # Validate
            if (epoch + 1) % self.config.TRAIN.VAL_INTERVAL == 0:
                ap, pck = self.validate(epoch)
                
                # Save checkpoint
                is_best = ap > self.best_ap
                if is_best:
                    self.best_ap = ap
                
                self.save_checkpoint(epoch, ap, is_best)
            
            # Update learning rate
            self.scheduler.step()


def main():
    args = parse_args()
    
    # Load configuration
    config = get_config(args.config)
    config.DATA.DATA_DIR = args.data_dir
    config.OUTPUT_DIR = args.output_dir
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config.__dict__, f, indent=4)
    
    # Create datasets
    train_dataset = PreemieCocoDataset(
        config,
        ann_file=os.path.join(args.data_dir, 'annotations/train.json'),
        img_dir=os.path.join(args.data_dir, 'images/train'),
        is_train=True
    )
    
    val_dataset = PreemieCocoDataset(
        config,
        ann_file=os.path.join(args.data_dir, 'annotations/val.json'),
        img_dir=os.path.join(args.data_dir, 'images/val'),
        is_train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=config.TRAIN.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=config.TEST.NUM_WORKERS,
        pin_memory=True
    )
    
    # Create model
    model = PoseHighResolutionNet(config)
    
    # Load checkpoint if resume
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded checkpoint from {args.resume}')
    
    # Create trainer and start training
    trainer = Trainer(config, model, train_loader, val_loader, device)
    
    print(f'Starting training on device: {device}')
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    
    trainer.train()


if __name__ == '__main__':
    main()
