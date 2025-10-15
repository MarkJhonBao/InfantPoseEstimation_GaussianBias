"""
Inference script for Preterm Infant Pose Estimation
Supports single image, batch, and video inference
"""
import torch
import cv2
import numpy as np
import argparse
import os
import json
from tqdm import tqdm

from models.pose_hrnet import PoseHighResolutionNet
from utils.postprocess import fused_decode, temporal_smoothing
from utils.visualization import draw_keypoints, create_video_with_pose
from config import get_config


def parse_args():
    parser = argparse.ArgumentParser(description='Inference for Preterm Infant Pose Estimation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='path to input image')
    parser.add_argument('--video', type=str, default=None,
                        help='path to input video')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='directory of input images')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='directory to save results')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize results')
    parser.add_argument('--save_json', action='store_true',
                        help='save predictions to JSON')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU id')
    return parser.parse_args()


class InferenceEngine:
    """Inference engine for pose estimation"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.config = checkpoint.get('config', get_config())
        
        # Build model
        self.model = PoseHighResolutionNet(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        
        # Image preprocessing
        self.image_size = self.config.MODEL.IMAGE_SIZE
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Device: {device}")
    
    def preprocess_image(self, image):
        """
        Preprocess image for model input
        
        Args:
            image: (H, W, 3) RGB image
        Returns:
            image_tensor: (1, 3, H, W) normalized tensor
            meta: dict with preprocessing metadata
        """
        h, w = image.shape[:2]
        
        # Simple detection: assume single infant in center
        # In production, use person detector first
        bbox = np.array([w*0.1, h*0.1, w*0.8, h*0.8], dtype=np.float32)
        center = np.array([bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2])
        scale = np.array([bbox[2], bbox[3]])
        
        # Crop and resize
        x1, y1 = int(bbox[0]), int(bbox[1])
        x2, y2 = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        cropped = image[y1:y2, x1:x2]
        
        resized = cv2.resize(cropped, tuple(self.image_size))
        
        # Normalize
        image_normalized = (resized / 255.0 - self.mean) / self.std
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0)
        
        meta = {
            'bbox': bbox,
            'center': center,
            'scale': scale,
            'original_shape': (h, w)
        }
        
        return image_tensor, meta
    
    def predict_single(self, image):
        """
        Run inference on single image
        
        Args:
            image: (H, W, 3) RGB image
        Returns:
            predictions: dict with keypoints and confidence
        """
        # Preprocess
        image_tensor, meta = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # Decode predictions
        preds, maxvals = fused_decode(
            outputs['heatmaps'],
            outputs.get('coords', None),
            torch.from_numpy(meta['center']).unsqueeze(0),
            torch.from_numpy(meta['scale']).unsqueeze(0)
        )
        
        # Convert to numpy
        keypoints = preds[0].cpu().numpy()
        confidence = maxvals[0].cpu().numpy()
        
        # Transform back to original image coordinates
        bbox = meta['bbox']
        keypoints[:, 0] = keypoints[:, 0] / self.image_size[0] * bbox[2] + bbox[0]
        keypoints[:, 1] = keypoints[:, 1] / self.image_size[1] * bbox[3] + bbox[1]
        
        predictions = {
            'keypoints': keypoints,
            'confidence': confidence,
            'bbox': bbox
        }
        
        return predictions
    
    def predict_video(self, video_path, output_path=None, smooth=True):
        """
        Run inference on video with temporal smoothing
        
        Args:
            video_path: path to input video
            output_path: path to save output video
            smooth: whether to apply temporal smoothing
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Collect all predictions first for temporal smoothing
        all_keypoints = []
        all_frames = []
        
        pbar = tqdm(total=total_frames, desc="Extracting poses")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictions = self.predict_single(frame_rgb)
            
            all_keypoints.append(predictions['keypoints'])
            all_frames.append(frame)
            
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        # Apply temporal smoothing
        if smooth and len(all_keypoints) > 1:
            print("Applying temporal smoothing...")
            keypoints_array = np.array(all_keypoints)
            smoothed_keypoints = temporal_smoothing(
                torch.from_numpy(keypoints_array),
                window_size=5
            ).numpy()
            all_keypoints = list(smoothed_keypoints)
        
        # Save video with pose overlay
        if output_path:
            print(f"Saving output video to {output_path}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame, keypoints in tqdm(zip(all_frames, all_keypoints), 
                                         total=len(all_frames),
                                         desc="Rendering video"):
                # Draw keypoints on frame
                vis_frame = draw_keypoints(frame, keypoints)
                out.write(vis_frame)
            
            out.release()
            print(f"Video saved successfully")
        
        return all_keypoints
    
    def predict_batch(self, image_dir, output_dir, visualize=True, save_json=True):
        """
        Run inference on directory of images
        
        Args:
            image_dir: directory containing images
            output_dir: directory to save results
            visualize: whether to save visualizations
            save_json: whether to save predictions to JSON
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get image files
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Found {len(image_files)} images in {image_dir}")
        
        all_predictions = {}
        
        for img_file in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(image_dir, img_file)
            image = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Predict
            predictions = self.predict_single(image_rgb)
            
            # Save results
            img_id = os.path.splitext(img_file)[0]
            all_predictions[img_id] = {
                'keypoints': predictions['keypoints'].tolist(),
                'confidence': predictions['confidence'].tolist()
            }
            
            # Visualize
            if visualize:
                vis_image = draw_keypoints(image, predictions['keypoints'])
                vis_path = os.path.join(output_dir, f"{img_id}_pose.jpg")
                cv2.imwrite(vis_path, vis_image)
        
        # Save JSON
        if save_json:
            json_path = os.path.join(output_dir, 'predictions.json')
            with open(json_path, 'w') as f:
                json.dump(all_predictions, f, indent=4)
            print(f"Predictions saved to {json_path}")
        
        return all_predictions


def main():
    args = parse_args()
    
    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create inference engine
    engine = InferenceEngine(args.checkpoint, device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run inference based on input type
    if args.image:
        # Single image
        print(f"Processing image: {args.image}")
        image = cv2.imread(args.image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        predictions = engine.predict_single(image_rgb)
        
        # Visualize
        if args.visualize:
            vis_image = draw_keypoints(image, predictions['keypoints'])
            output_path = os.path.join(args.output_dir, 'result.jpg')
            cv2.imwrite(output_path, vis_image)
            print(f"Result saved to {output_path}")
        
        # Save JSON
        if args.save_json:
            json_path = os.path.join(args.output_dir, 'predictions.json')
            with open(json_path, 'w') as f:
                json.dump({
                    'keypoints': predictions['keypoints'].tolist(),
                    'confidence': predictions['confidence'].tolist()
                }, f, indent=4)
    
    elif args.video:
        # Video
        output_video = os.path.join(args.output_dir, 'output_video.mp4')
        keypoints_sequence = engine.predict_video(args.video, output_video)
        
        # Save predictions
        if args.save_json:
            json_path = os.path.join(args.output_dir, 'video_predictions.json')
            predictions_list = [kp.tolist() for kp in keypoints_sequence]
            with open(json_path, 'w') as f:
                json.dump({'frames': predictions_list}, f, indent=4)
    
    elif args.image_dir:
        # Batch processing
        engine.predict_batch(
            args.image_dir,
            args.output_dir,
            visualize=args.visualize,
            save_json=args.save_json
        )
    
    else:
        print("Please specify --image, --video, or --image_dir")
        return
    
    print("Inference completed!")


if __name__ == '__main__':
    main()
