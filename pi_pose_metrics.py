"""
Evaluation metrics for preterm infant pose estimation
Includes AP (Average Precision) and PCK (Percentage of Correct Keypoints)
"""
import numpy as np
from collections import defaultdict
from pycocotools.cocoeval import COCOeval
import json


def calculate_pck(predictions, bboxes, image_ids, dataset, threshold=0.2):
    """
    Calculate PCK (Percentage of Correct Keypoints)
    
    A keypoint is considered correct if the distance between predicted and
    ground truth is within threshold * bbox_diagonal
    
    Args:
        predictions: list of (N, K, 2) predicted coordinates
        bboxes: list of (N, 4) bounding boxes [x, y, w, h]
        image_ids: list of image IDs
        dataset: dataset object with ground truth
        threshold: PCK threshold (default 0.2)
    
    Returns:
        pck: PCK score
    """
    all_preds = np.concatenate(predictions, axis=0)
    all_boxes = np.concatenate(bboxes, axis=0)
    all_ids = np.concatenate(image_ids, axis=0)
    
    num_samples = len(all_preds)
    num_joints = all_preds.shape[1]
    
    correct = np.zeros((num_samples, num_joints))
    total = np.zeros((num_samples, num_joints))
    
    for i in range(num_samples):
        img_id = all_ids[i]
        
        # Get ground truth
        ann_ids = dataset.coco.getAnnIds(imgIds=int(img_id))
        if len(ann_ids) == 0:
            continue
        
        ann = dataset.coco.loadAnns(ann_ids)[0]
        gt_keypoints = np.array(ann['keypoints']).reshape(-1, 3)
        gt_joints = gt_keypoints[:, :2]
        gt_vis = gt_keypoints[:, 2]
        
        # Calculate bbox diagonal
        bbox = all_boxes[i]
        bbox_diag = np.sqrt(bbox[2]**2 + bbox[3]**2)
        threshold_dist = threshold * bbox_diag
        
        # Calculate distances
        for j in range(num_joints):
            if gt_vis[j] > 0:  # Only evaluate visible joints
                total[i, j] = 1
                
                pred_joint = all_preds[i, j]
                gt_joint = gt_joints[j]
                
                distance = np.linalg.norm(pred_joint - gt_joint)
                
                if distance <= threshold_dist:
                    correct[i, j] = 1
    
    # Calculate PCK per joint
    pck_per_joint = correct.sum(axis=0) / (total.sum(axis=0) + 1e-8)
    
    # Overall PCK
    pck = correct.sum() / (total.sum() + 1e-8)
    
    return pck, pck_per_joint


def calculate_ap(predictions, bboxes, image_ids, dataset, oks_threshold=0.5):
    """
    Calculate AP (Average Precision) using OKS (Object Keypoint Similarity)
    
    Args:
        predictions: list of (N, K, 2) predicted coordinates
        bboxes: list of (N, 4) bounding boxes
        image_ids: list of image IDs
        dataset: dataset object with COCO API
        oks_threshold: OKS threshold for positive detection
    
    Returns:
        ap: Average Precision
    """
    # Convert predictions to COCO format
    coco_results = []
    
    all_preds = np.concatenate(predictions, axis=0)
    all_boxes = np.concatenate(bboxes, axis=0)
    all_ids = np.concatenate(image_ids, axis=0)
    
    for i in range(len(all_preds)):
        img_id = int(all_ids[i])
        bbox = all_boxes[i].tolist()
        keypoints = all_preds[i].flatten().tolist()
        
        # Add visibility flags (all visible for now)
        keypoints_with_vis = []
        for j in range(0, len(keypoints), 2):
            keypoints_with_vis.extend([keypoints[j], keypoints[j+1], 2])
        
        result = {
            'image_id': img_id,
            'category_id': 1,
            'keypoints': keypoints_with_vis,
            'score': 1.0  # Confidence score
        }
        coco_results.append(result)
    
    if len(coco_results) == 0:
        return 0.0
    
    # Use COCOeval to calculate AP
    coco_dt = dataset.coco.loadRes(coco_results)
    coco_eval = COCOeval(dataset.coco, coco_dt, 'keypoints')
    coco_eval.params.imgIds = list(set(all_ids.tolist()))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # AP is the first metric
    ap = coco_eval.stats[0]
    
    return ap


def calculate_oks(pred_keypoints, gt_keypoints, bbox, sigmas=None):
    """
    Calculate OKS (Object Keypoint Similarity) for a single sample
    
    OKS = Σ exp(-d²/(2s²k²)) δ(v>0) / Σ δ(v>0)
    
    Where:
    - d: distance between predicted and ground truth keypoint
    - s: scale = sqrt(bbox_area)
    - k: per-keypoint constant (default from COCO)
    - v: visibility flag
    
    Args:
        pred_keypoints: (K, 2) predicted keypoints
        gt_keypoints: (K, 3) ground truth keypoints [x, y, v]
        bbox: (4,) bounding box [x, y, w, h]
        sigmas: (K,) per-keypoint sigmas
    """
    num_joints = len(pred_keypoints)
    
    if sigmas is None:
        # Default COCO sigmas (adjust for preterm infants if needed)
        sigmas = np.array([
            0.026, 0.025, 0.025,  # nose, eyes
            0.035, 0.035,          # ears
            0.079, 0.079,          # shoulders
            0.072, 0.072,          # elbows
            0.062, 0.062,          # wrists
            0.107, 0.107           # hips
        ])
    
    # Calculate scale from bbox
    bbox_area = bbox[2] * bbox[3]
    scale = np.sqrt(bbox_area)
    
    # Calculate distances
    gt_coords = gt_keypoints[:, :2]
    gt_vis = gt_keypoints[:, 2]
    
    distances = np.linalg.norm(pred_keypoints - gt_coords, axis=1)
    
    # Calculate OKS
    oks_per_joint = np.exp(-distances**2 / (2 * scale**2 * sigmas**2))
    oks_per_joint = oks_per_joint * (gt_vis > 0)
    
    oks = oks_per_joint.sum() / ((gt_vis > 0).sum() + 1e-8)
    
    return oks


def calculate_per_joint_accuracy(predictions, bboxes, image_ids, dataset):
    """
    Calculate accuracy for each individual joint
    
    Returns:
        accuracy_dict: dictionary mapping joint names to accuracies
    """
    # Get joint names from dataset
    if hasattr(dataset, 'joint_names'):
        joint_names = dataset.joint_names
    else:
        joint_names = [
            'nose', 'left_eye', 'right_eye',
            'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_hip', 'right_hip'
        ]
    
    pck, pck_per_joint = calculate_pck(predictions, bboxes, image_ids, dataset)
    
    accuracy_dict = {}
    for i, name in enumerate(joint_names):
        accuracy_dict[name] = pck_per_joint[i]
    
    return accuracy_dict


def calculate_temporal_consistency(predictions_sequence):
    """
    Calculate temporal consistency for video sequences
    Measures smoothness of trajectories
    
    Args:
        predictions_sequence: (T, K, 2) sequence of predictions over time
    
    Returns:
        consistency_score: lower is better (0 = perfectly smooth)
    """
    if len(predictions_sequence) < 2:
        return 0.0
    
    # Calculate frame-to-frame velocity
    velocities = np.diff(predictions_sequence, axis=0)
    
    # Calculate acceleration (change in velocity)
    accelerations = np.diff(velocities, axis=0)
    
    # Consistency score = average magnitude of acceleration
    consistency_score = np.mean(np.linalg.norm(accelerations, axis=2))
    
    return consistency_score


def calculate_movement_amplitude(predictions_sequence):
    """
    Calculate movement amplitude for each joint
    Useful for assessing motor activity levels
    """
    if len(predictions_sequence) < 2:
        return np.zeros(predictions_sequence.shape[1])
    
    # Calculate range of motion for each joint
    min_pos = predictions_sequence.min(axis=0)
    max_pos = predictions_sequence.max(axis=0)
    
    amplitude = np.linalg.norm(max_pos - min_pos, axis=1)
    
    return amplitude


def evaluate_model(model, dataloader, device, dataset):
    """
    Complete evaluation pipeline
    
    Returns:
        metrics: dictionary with all evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_boxes = []
    all_ids = []
    
    import torch
    from utils.postprocess import fused_decode
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Decode predictions
            preds, maxvals = fused_decode(
                outputs['heatmaps'],
                outputs.get('coords', None),
                batch['center'],
                batch['scale']
            )
            
            all_preds.append(preds.cpu().numpy())
            all_boxes.append(batch['bbox'].cpu().numpy())
            all_ids.append(batch['image_id'].cpu().numpy())
    
    # Calculate metrics
    ap = calculate_ap(all_preds, all_boxes, all_ids, dataset)
    pck, pck_per_joint = calculate_pck(all_preds, all_boxes, all_ids, dataset)
    per_joint_acc = calculate_per_joint_accuracy(all_preds, all_boxes, all_ids, dataset)
    
    metrics = {
        'AP': ap,
        'PCK@0.2': pck,
        'per_joint_PCK': pck_per_joint,
        'per_joint_accuracy': per_joint_acc
    }
    
    return metrics


def print_metrics(metrics):
    """Pretty print evaluation metrics"""
    print("\n" + "="*80)
    print("EVALUATION METRICS")
    print("="*80)
    
    print(f"AP (Average Precision): {metrics['AP']:.4f}")
    print(f"PCK@0.2: {metrics['PCK@0.2']:.4f}")
    
    if 'per_joint_accuracy' in metrics:
        print("\nPer-Joint Accuracy:")
        print("-"*80)
        for joint_name, accuracy in metrics['per_joint_accuracy'].items():
            print(f"  {joint_name:20s}: {accuracy:.4f}")
    
    print("="*80 + "\n")


def save_metrics(metrics, output_path):
    """Save metrics to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    metrics_json = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_json[key] = value.tolist()
        elif isinstance(value, dict):
            metrics_json[key] = {k: float(v) if isinstance(v, np.ndarray) else v 
                                for k, v in value.items()}
        else:
            metrics_json[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value
    
    with open(output_path, 'w') as f:
        json.dump(metrics_json, f, indent=4)
    
    print(f"Metrics saved to {output_path}")
