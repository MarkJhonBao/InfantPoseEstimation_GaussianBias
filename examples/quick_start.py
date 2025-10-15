"""
Quick Start Example for Preterm Infant Pose Estimation

This script demonstrates:
1. Loading a trained model
2. Running inference on a single image
3. Visualizing results
4. Analyzing movement from video
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pose_hrnet import PoseHighResolutionNet
from utils.visualization import draw_keypoints, plot_movement_trajectory
from utils.postprocess import fused_decode, temporal_smoothing
from config import get_config


def example_single_image_inference():
    """
    Example 1: Run inference on a single image
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Image Inference")
    print("="*80)
    
    # Load configuration
    config = get_config('configs/default.yaml')
    
    # Create model
    model = PoseHighResolutionNet(config)
    
    # Load pretrained weights (if available)
    checkpoint_path = 'checkpoints/model_best.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded pretrained weights")
    else:
        print("⚠ No pretrained weights found. Using random initialization.")
    
    model.eval()
    
    # Load and preprocess image
    image_path = 'examples/sample_image.jpg'
    
    if not os.path.exists(image_path):
        print(f"⚠ Sample image not found at {image_path}")
        print("  Creating a dummy image for demonstration...")
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(image_path, dummy_image)
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    resized = cv2.resize(image_rgb, tuple(config.MODEL.IMAGE_SIZE))
    normalized = (resized / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    image_tensor = torch.from_numpy(normalized).permute(2, 0, 1).float().unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Decode predictions
    preds, maxvals = fused_decode(
        outputs['heatmaps'],
        outputs.get('coords', None)
    )
    
    keypoints = preds[0].cpu().numpy()
    confidence = maxvals[0].cpu().numpy()
    
    # Scale back to original image size
    h, w = image.shape[:2]
    keypoints[:, 0] = keypoints[:, 0] / config.MODEL.IMAGE_SIZE[0] * w
    keypoints[:, 1] = keypoints[:, 1] / config.MODEL.IMAGE_SIZE[1] * h
    
    # Visualize
    vis_image = draw_keypoints(image, keypoints, confidence)
    
    output_path = 'examples/output_single_image.jpg'
    cv2.imwrite(output_path, vis_image)
    
    print(f"✓ Inference complete")
    print(f"✓ Result saved to: {output_path}")
    print(f"  Detected {(confidence > 0.3).sum()} keypoints with confidence > 0.3")
    
    return keypoints, confidence


def example_video_analysis():
    """
    Example 2: Analyze infant movement from video
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Video Movement Analysis")
    print("="*80)
    
    # Simulate a video sequence (in practice, load from actual video)
    print("⚠ This is a simulation. In practice, load frames from actual video.")
    
    # Generate dummy trajectory (simulating infant limb movement)
    n_frames = 100
    n_keypoints = 13
    
    # Simulate smooth movement with some noise
    t = np.linspace(0, 4*np.pi, n_frames)
    base_x = 200 + 50 * np.sin(t)
    base_y = 200 + 30 * np.cos(t)
    
    keypoints_sequence = []
    for i in range(n_frames):
        frame_keypoints = np.zeros((n_keypoints, 2))
        
        # Simulate keypoint positions (simplified)
        for j in range(n_keypoints):
            offset_x = np.random.randn() * 5  # Add noise
            offset_y = np.random.randn() * 5
            
            frame_keypoints[j, 0] = base_x[i] + j * 20 + offset_x
            frame_keypoints[j, 1] = base_y[i] + j * 15 + offset_y
        
        keypoints_sequence.append(frame_keypoints)
    
    keypoints_sequence = torch.from_numpy(np.array(keypoints_sequence)).float()
    
    print(f"✓ Generated sequence: {n_frames} frames, {n_keypoints} keypoints")
    
    # Apply temporal smoothing
    smoothed_sequence = temporal_smoothing(keypoints_sequence, window_size=5)
    
    print("✓ Applied temporal smoothing")
    
    # Plot movement trajectories
    fig = plot_movement_trajectory(
        smoothed_sequence.numpy(),
        joint_indices=[0, 5, 6, 9, 10],  # nose, shoulders, wrists
        joint_names=['Nose', 'L Shoulder', 'R Shoulder', 'L Wrist', 'R Wrist']
    )
    
    output_path = 'examples/movement_trajectory.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Trajectory plot saved to: {output_path}")
    
    # Calculate movement statistics
    from utils.metrics import calculate_movement_amplitude, calculate_temporal_consistency
    
    amplitude = calculate_movement_amplitude(smoothed_sequence.numpy())
    consistency = calculate_temporal_consistency(smoothed_sequence.numpy())
    
    print("\nMovement Statistics:")
    print(f"  Average amplitude: {amplitude.mean():.2f} pixels")
    print(f"  Temporal consistency: {consistency:.2f} (lower is smoother)")
    
    return smoothed_sequence


def example_batch_processing():
    """
    Example 3: Batch process multiple images
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Batch Processing")
    print("="*80)
    
    # Create dummy batch
    batch_size = 4
    n_keypoints = 13
    
    print(f"Processing batch of {batch_size} images...")
    
    # Simulate batch predictions
    predictions = []
    
    for i in range(batch_size):
        # Generate random keypoints (in practice, these come from model)
        keypoints = np.random.rand(n_keypoints, 2) * 400 + 100
        confidence = np.random.rand(n_keypoints, 1) * 0.5 + 0.5
        
        predictions.append({
            'keypoints': keypoints,
            'confidence': confidence,
            'image_id': i
        })
    
    print(f"✓ Processed {len(predictions)} images")
    
    # Calculate average confidence
    avg_conf = np.mean([p['confidence'].mean() for p in predictions])
    print(f"  Average confidence: {avg_conf:.3f}")
    
    return predictions


def example_clinical_analysis():
    """
    Example 4: Clinical movement analysis
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Clinical Movement Analysis")
    print("="*80)
    
    # Generate sample movement data
    n_frames = 200
    n_keypoints = 13
    
    # Simulate different movement patterns
    t = np.linspace(0, 10, n_frames)
    
    # Normal movement
    normal_amplitude = 30
    normal_freq = 0.5
    
    # Simulate left and right arm movements
    left_wrist_x = 200 + normal_amplitude * np.sin(2*np.pi*normal_freq*t)
    right_wrist_x = 400 + normal_amplitude * np.sin(2*np.pi*normal_freq*t + np.pi/4)
    
    left_wrist_y = 300 + normal_amplitude * np.cos(2*np.pi*normal_freq*t)
    right_wrist_y = 300 + normal_amplitude * np.cos(2*np.pi*normal_freq*t + np.pi/4)
    
    print("Movement Pattern Analysis:")
    print(f"  Left wrist amplitude: {normal_amplitude:.1f} pixels")
    print(f"  Right wrist amplitude: {normal_amplitude:.1f} pixels")
    print(f"  Movement frequency: {normal_freq:.2f} Hz")
    
    # Check for asymmetry
    left_range = left_wrist_x.max() - left_wrist_x.min()
    right_range = right_wrist_x.max() - right_wrist_x.min()
    asymmetry_ratio = abs(left_range - right_range) / max(left_range, right_range)
    
    print(f"\nAsymmetry Analysis:")
    print(f"  Left range of motion: {left_range:.1f} pixels")
    print(f"  Right range of motion: {right_range:.1f} pixels")
    print(f"  Asymmetry ratio: {asymmetry_ratio:.3f}")
    
    if asymmetry_ratio > 0.2:
        print("  ⚠ Significant asymmetry detected")
    else:
        print("  ✓ Movement is symmetric")
    
    # Activity level assessment
    total_movement = np.sqrt(
        np.diff(left_wrist_x)**2 + np.diff(left_wrist_y)**2 +
        np.diff(right_wrist_x)**2 + np.diff(right_wrist_y)**2
    ).sum()
    
    print(f"\nActivity Level:")
    print(f"  Total movement: {total_movement:.1f} pixels")
    
    activity_level = "Normal"
    if total_movement < 1000:
        activity_level = "Low"
    elif total_movement > 5000:
        activity_level = "High"
    
    print(f"  Assessment: {activity_level} activity")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("PRETERM INFANT POSE ESTIMATION - QUICK START EXAMPLES")
    print("="*80)
    
    # Create examples directory
    os.makedirs('examples', exist_ok=True)
    
    try:
        # Example 1: Single image inference
        keypoints, confidence = example_single_image_inference()
        
        # Example 2: Video analysis
        trajectory = example_video_analysis()
        
        # Example 3: Batch processing
        batch_results = example_batch_processing()
        
        # Example 4: Clinical analysis
        example_clinical_analysis()
        
        print("\n" + "="*80)
        print("✓ All examples completed successfully!")
        print("="*80)
        print("\nGenerated files:")
        print("  - examples/output_single_image.jpg")
        print("  - examples/movement_trajectory.png")
        print("\nFor more examples, see the documentation.")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("Make sure all dependencies are installed and configurations are set correctly.")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
