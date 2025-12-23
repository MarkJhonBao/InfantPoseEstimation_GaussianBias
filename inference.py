"""
Inference Script for Human Pose Estimation
Single image and batch inference with visualization
"""

import os
import sys
import argparse
import time
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_config
from models import build_model
from utils import draw_skeleton, draw_heatmaps, save_visualization, COCO_SKELETON


class PoseInference:
    """Pose estimation inference class.
    
    Args:
        checkpoint: Path to model checkpoint.
        device: Device to run inference on.
        flip_test: Whether to use flip test augmentation.
    """
    
    def __init__(
        self,
        checkpoint: str,
        device: str = 'cuda',
        flip_test: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.flip_test = flip_test
        
        # Load config
        self.cfg = get_config()
        
        # Build model
        self.model = build_model(self.cfg)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load checkpoint
        if checkpoint and os.path.isfile(checkpoint):
            ckpt = torch.load(checkpoint, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            print(f'Loaded checkpoint: {checkpoint}')
        
        # Preprocessing params
        self.input_size = self.cfg.data.input_size  # (w, h)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Flip pairs
        self.flip_pairs = self.cfg.data.flip_pairs
    
    def preprocess(
        self,
        img: np.ndarray,
        bbox: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Preprocess image for inference.
        
        Args:
            img: Input image (H, W, 3) in BGR format.
            bbox: Optional bounding box (x1, y1, x2, y2).
            
        Returns:
            input_tensor: Preprocessed tensor.
            center: Bbox center.
            scale: Bbox scale.
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get bbox
        if bbox is None:
            # Use full image
            h, w = img.shape[:2]
            bbox = np.array([0, 0, w, h])
        
        x1, y1, x2, y2 = bbox
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        scale = np.array([x2 - x1, y2 - y1]) * 1.25
        
        # Affine transform
        trans = self._get_affine_transform(center, scale, self.input_size)
        img_cropped = cv2.warpAffine(
            img, trans,
            (int(self.input_size[0]), int(self.input_size[1])),
            flags=cv2.INTER_LINEAR
        )
        
        # Normalize
        img_cropped = img_cropped.astype(np.float32) / 255.0
        img_cropped = (img_cropped - self.mean) / self.std
        
        # To tensor
        input_tensor = torch.from_numpy(
            img_cropped.transpose(2, 0, 1)
        ).float().unsqueeze(0)
        
        return input_tensor.to(self.device), center, scale
    
    def _get_affine_transform(
        self,
        center: np.ndarray,
        scale: np.ndarray,
        output_size: np.ndarray,
    ) -> np.ndarray:
        """Get affine transform matrix."""
        src_w = scale[0]
        dst_w, dst_h = output_size
        
        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        
        src_dir = np.array([0, src_w * -0.5])
        dst_dir = np.array([0, dst_w * -0.5])
        
        src[0, :] = center
        src[1, :] = center + src_dir
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        
        src[2, :] = self._get_3rd_point(src[0, :], src[1, :])
        dst[2, :] = self._get_3rd_point(dst[0, :], dst[1, :])
        
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return trans
    
    def _get_3rd_point(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Get third point for affine transform."""
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)
    
    def postprocess(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        center: np.ndarray,
        scale: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess predictions to original image coordinates.
        
        Args:
            keypoints: Predicted keypoints in heatmap space.
            scores: Keypoint scores.
            center: Bbox center.
            scale: Bbox scale.
            
        Returns:
            keypoints: Keypoints in original image space.
            scores: Keypoint scores.
        """
        # Scale from heatmap to input size
        heatmap_size = self.cfg.data.heatmap_size
        scale_x = self.input_size[0] / heatmap_size[0]
        scale_y = self.input_size[1] / heatmap_size[1]
        
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y
        
        # Transform to original image coordinates
        for k in range(len(keypoints)):
            keypoints[k, 0] = keypoints[k, 0] / self.input_size[0] * scale[0] + center[0] - scale[0] / 2
            keypoints[k, 1] = keypoints[k, 1] / self.input_size[1] * scale[1] + center[1] - scale[1] / 2
        
        return keypoints, scores
    
    @torch.no_grad()
    def predict(
        self,
        img: np.ndarray,
        bbox: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference on single image.
        
        Args:
            img: Input image (H, W, 3) in BGR format.
            bbox: Optional bounding box (x1, y1, x2, y2).
            
        Returns:
            keypoints: Predicted keypoints (K, 2).
            scores: Keypoint scores (K,).
        """
        # Preprocess
        input_tensor, center, scale = self.preprocess(img, bbox)
        
        # Inference
        if self.flip_test:
            keypoints, scores = self.model.inference(
                input_tensor, flip=True, flip_pairs=self.flip_pairs
            )
        else:
            outputs = self.model(input_tensor)
            keypoints, scores = self.model.decode_heatmaps(outputs['heatmaps'])
        
        keypoints = keypoints[0].cpu().numpy()
        scores = scores[0].cpu().numpy()
        
        # Postprocess
        keypoints, scores = self.postprocess(keypoints, scores, center, scale)
        
        return keypoints, scores
    
    def predict_batch(
        self,
        imgs: List[np.ndarray],
        bboxes: Optional[List[np.ndarray]] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Run inference on batch of images.
        
        Args:
            imgs: List of input images.
            bboxes: Optional list of bounding boxes.
            
        Returns:
            List of (keypoints, scores) tuples.
        """
        results = []
        for i, img in enumerate(imgs):
            bbox = bboxes[i] if bboxes else None
            keypoints, scores = self.predict(img, bbox)
            results.append((keypoints, scores))
        return results
    
    def visualize(
        self,
        img: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        score_threshold: float = 0.3,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """Visualize predictions on image.
        
        Args:
            img: Input image.
            keypoints: Predicted keypoints.
            scores: Keypoint scores.
            score_threshold: Minimum score to draw.
            output_path: Optional path to save result.
            
        Returns:
            Visualization image.
        """
        vis_img = draw_skeleton(
            img, keypoints, scores,
            score_threshold=score_threshold,
            skeleton=COCO_SKELETON,
        )
        
        if output_path:
            cv2.imwrite(output_path, vis_img)
        
        return vis_img


def detect_persons(img: np.ndarray) -> List[np.ndarray]:
    """Simple person detection using OpenCV's HOG detector.
    
    For production, use a proper object detector like YOLO, Faster R-CNN, etc.
    
    Args:
        img: Input image.
        
    Returns:
        List of bounding boxes.
    """
    # Use full image as single detection (placeholder)
    # In production, integrate with a person detector
    h, w = img.shape[:2]
    return [np.array([0, 0, w, h])]


def main(args):
    """Main inference function."""
    print('Initializing pose estimator...')
    
    # Initialize inference
    pose_estimator = PoseInference(
        checkpoint=args.checkpoint,
        device=args.device,
        flip_test=not args.no_flip,
    )
    
    # Process input
    if os.path.isfile(args.input):
        # Single image
        print(f'Processing image: {args.input}')
        
        img = cv2.imread(args.input)
        if img is None:
            raise ValueError(f'Failed to load image: {args.input}')
        
        # Detect persons (or use provided bbox)
        if args.bbox:
            bboxes = [np.array(args.bbox)]
        else:
            bboxes = detect_persons(img)
        
        # Run inference
        start_time = time.time()
        
        for i, bbox in enumerate(bboxes):
            keypoints, scores = pose_estimator.predict(img, bbox)
            
            # Visualize
            img = draw_skeleton(img, keypoints, scores, score_threshold=args.threshold)
        
        inference_time = time.time() - start_time
        print(f'Inference time: {inference_time * 1000:.2f} ms')
        
        # Save result
        if args.output:
            output_path = args.output
        else:
            base, ext = os.path.splitext(args.input)
            output_path = f'{base}_result{ext}'
        
        cv2.imwrite(output_path, img)
        print(f'Result saved to: {output_path}')
        
        # Print keypoints
        if args.verbose:
            print('\nPredicted keypoints:')
            keypoint_names = pose_estimator.cfg.data.keypoint_names
            for i, (name, kpt, score) in enumerate(zip(keypoint_names, keypoints, scores)):
                print(f'  {i:2d}. {name:15s}: ({kpt[0]:7.2f}, {kpt[1]:7.2f}) score={score:.3f}')
    
    elif os.path.isdir(args.input):
        # Directory of images
        print(f'Processing directory: {args.input}')
        
        output_dir = args.output or os.path.join(args.input, 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(args.input) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        total_time = 0
        for filename in image_files:
            img_path = os.path.join(args.input, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f'  Skipping {filename}: failed to load')
                continue
            
            # Detect persons
            bboxes = detect_persons(img)
            
            # Run inference
            start_time = time.time()
            for bbox in bboxes:
                keypoints, scores = pose_estimator.predict(img, bbox)
                img = draw_skeleton(img, keypoints, scores, score_threshold=args.threshold)
            total_time += time.time() - start_time
            
            # Save result
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img)
            print(f'  Processed: {filename}')
        
        print(f'\nProcessed {len(image_files)} images')
        print(f'Average inference time: {total_time / len(image_files) * 1000:.2f} ms')
        print(f'Results saved to: {output_dir}')
    
    else:
        raise ValueError(f'Invalid input: {args.input}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose Estimation Inference')
    parser.add_argument('--input', type=str, required=True, 
                        help='Input image or directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Score threshold for visualization')
    parser.add_argument('--bbox', type=float, nargs=4, default=None,
                        help='Bounding box (x1 y1 x2 y2)')
    parser.add_argument('--no_flip', action='store_true',
                        help='Disable flip test')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed results')
    
    args = parser.parse_args()
    main(args)
