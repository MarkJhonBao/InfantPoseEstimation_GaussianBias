"""
Evaluation Metrics for Pose Estimation
Including OKS-based AP computation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


class COCOEvaluator:
    """COCO-style pose estimation evaluator.
    
    Args:
        ann_file: Path to COCO annotation file.
        oks_sigmas: OKS sigma values for each keypoint.
    """
    
    # Default COCO OKS sigmas
    DEFAULT_OKS_SIGMAS = np.array([
        0.026,  # nose
        0.025,  # left_eye
        0.025,  # right_eye
        0.035,  # left_ear
        0.035,  # right_ear
        0.079,  # left_shoulder
        0.079,  # right_shoulder
        0.072,  # left_elbow
        0.072,  # right_elbow
        0.062,  # left_wrist
        0.062,  # right_wrist
        0.107,  # left_hip
        0.107,  # right_hip
        0.087,  # left_knee
        0.087,  # right_knee
        0.089,  # left_ankle
        0.089,  # right_ankle
    ])
    
    def __init__(
        self,
        ann_file: Optional[str] = None,
        oks_sigmas: Optional[np.ndarray] = None,
        num_keypoints: int = 17,
    ):
        self.ann_file = ann_file
        self.oks_sigmas = oks_sigmas if oks_sigmas is not None else self.DEFAULT_OKS_SIGMAS
        self.num_keypoints = num_keypoints
        
        # OKS thresholds for AP computation
        self.oks_thresholds = np.linspace(0.5, 0.95, 10)
        
        # Storage for predictions and ground truths
        self.predictions = []
        self.reset()
    
    def reset(self):
        """Reset evaluator state."""
        self.predictions = []
    
    def update(
        self,
        pred_keypoints: np.ndarray,
        pred_scores: np.ndarray,
        image_ids: List[int],
        ann_ids: List[int],
        centers: np.ndarray,
        scales: np.ndarray,
        areas: np.ndarray,
        bboxes: np.ndarray,
    ):
        """Update with batch predictions.
        
        Args:
            pred_keypoints: Predicted keypoints (B, K, 2) in original image space.
            pred_scores: Keypoint scores (B, K).
            image_ids: Image IDs.
            ann_ids: Annotation IDs.
            centers: Bbox centers (B, 2).
            scales: Bbox scales (B, 2).
            areas: Bbox areas (B,).
            bboxes: Bounding boxes (B, 4).
        """
        batch_size = pred_keypoints.shape[0]
        
        for i in range(batch_size):
            # Format keypoints with visibility
            kpts = np.zeros((self.num_keypoints, 3))
            kpts[:, :2] = pred_keypoints[i]
            kpts[:, 2] = pred_scores[i]
            
            # Overall score (mean of valid keypoints)
            valid_mask = pred_scores[i] > 0
            if valid_mask.sum() > 0:
                score = pred_scores[i][valid_mask].mean()
            else:
                score = 0.0
            
            self.predictions.append({
                'image_id': int(image_ids[i]),
                'ann_id': int(ann_ids[i]),
                'keypoints': kpts.flatten().tolist(),
                'score': float(score),
                'area': float(areas[i]),
                'bbox': bboxes[i].tolist(),
            })
    
    def compute_oks(
        self,
        pred_kpts: np.ndarray,
        gt_kpts: np.ndarray,
        gt_vis: np.ndarray,
        area: float,
    ) -> float:
        """Compute Object Keypoint Similarity.
        
        Args:
            pred_kpts: Predicted keypoints (K, 2).
            gt_kpts: Ground truth keypoints (K, 2).
            gt_vis: Ground truth visibility (K,).
            area: Object area.
            
        Returns:
            OKS value.
        """
        # Compute distances
        dx = pred_kpts[:, 0] - gt_kpts[:, 0]
        dy = pred_kpts[:, 1] - gt_kpts[:, 1]
        d = dx ** 2 + dy ** 2
        
        # Normalize by area and sigma
        s = area
        k = self.oks_sigmas
        e = d / (2 * s * (k ** 2) + np.spacing(1))
        
        # Compute OKS
        valid = gt_vis > 0
        if valid.sum() == 0:
            return 0.0
        
        oks = np.sum(np.exp(-e[valid])) / valid.sum()
        
        return oks
    
    def evaluate(self, gt_annotations: Optional[List[Dict]] = None) -> Dict[str, float]:
        """Evaluate predictions.
        
        Args:
            gt_annotations: Ground truth annotations (if not using ann_file).
            
        Returns:
            Dictionary of metrics.
        """
        if len(self.predictions) == 0:
            return {'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0}
        
        # Load ground truth from COCO if ann_file provided
        if self.ann_file is not None:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            import json
            import tempfile
            import os
            
            # Save predictions to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(self.predictions, f)
                pred_file = f.name
            
            try:
                # Load COCO ground truth
                coco_gt = COCO(self.ann_file)
                
                # Load predictions
                coco_dt = coco_gt.loadRes(pred_file)
                
                # Run evaluation
                coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                
                # Extract metrics
                metrics = {
                    'AP': coco_eval.stats[0],
                    'AP50': coco_eval.stats[1],
                    'AP75': coco_eval.stats[2],
                    'AP_M': coco_eval.stats[3],
                    'AP_L': coco_eval.stats[4],
                    'AR': coco_eval.stats[5],
                    'AR50': coco_eval.stats[6],
                    'AR75': coco_eval.stats[7],
                    'AR_M': coco_eval.stats[8],
                    'AR_L': coco_eval.stats[9],
                }
            finally:
                os.unlink(pred_file)
            
            return metrics
        
        # Manual evaluation if gt_annotations provided
        elif gt_annotations is not None:
            return self._manual_evaluate(gt_annotations)
        
        else:
            raise ValueError("Either ann_file or gt_annotations must be provided")
    
    def _manual_evaluate(self, gt_annotations: List[Dict]) -> Dict[str, float]:
        """Manual OKS-based AP computation."""
        # Group predictions and GT by image
        pred_by_img = defaultdict(list)
        gt_by_img = defaultdict(list)
        
        for pred in self.predictions:
            pred_by_img[pred['image_id']].append(pred)
        
        for gt in gt_annotations:
            gt_by_img[gt['image_id']].append(gt)
        
        # Compute OKS for each threshold
        aps = []
        for thresh in self.oks_thresholds:
            tp = 0
            fp = 0
            fn = 0
            
            for img_id in gt_by_img:
                preds = pred_by_img[img_id]
                gts = gt_by_img[img_id]
                
                # Sort predictions by score
                preds = sorted(preds, key=lambda x: x['score'], reverse=True)
                
                matched = set()
                for pred in preds:
                    pred_kpts = np.array(pred['keypoints']).reshape(-1, 3)
                    best_oks = 0
                    best_gt_idx = -1
                    
                    for gt_idx, gt in enumerate(gts):
                        if gt_idx in matched:
                            continue
                        
                        gt_kpts = np.array(gt['keypoints']).reshape(-1, 3)
                        oks = self.compute_oks(
                            pred_kpts[:, :2],
                            gt_kpts[:, :2],
                            gt_kpts[:, 2],
                            gt['area']
                        )
                        
                        if oks > best_oks:
                            best_oks = oks
                            best_gt_idx = gt_idx
                    
                    if best_oks >= thresh and best_gt_idx >= 0:
                        tp += 1
                        matched.add(best_gt_idx)
                    else:
                        fp += 1
                
                fn += len(gts) - len(matched)
            
            # Compute precision at this threshold
            precision = tp / (tp + fp + 1e-10)
            aps.append(precision)
        
        return {
            'AP': np.mean(aps),
            'AP50': aps[0] if len(aps) > 0 else 0.0,
            'AP75': aps[5] if len(aps) > 5 else 0.0,
        }


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = '', fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class MetricLogger:
    """Logger for multiple metrics."""
    
    def __init__(self, delimiter: str = '  '):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)
    
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return super().__getattr__(attr)
    
    def __str__(self):
        entries = []
        for name, meter in self.meters.items():
            entries.append(f"{name}: {meter.avg:.4f}")
        return self.delimiter.join(entries)
    
    def summary(self) -> Dict[str, float]:
        return {name: meter.avg for name, meter in self.meters.items()}


# Import torch here to avoid circular import
import torch
