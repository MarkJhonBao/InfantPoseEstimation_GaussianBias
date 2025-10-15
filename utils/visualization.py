"""
Visualization utilities for preterm infant pose estimation
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches


# Preterm infant skeleton connections
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),      # nose to eyes
    (1, 3), (2, 4),      # eyes to ears
    (5, 6),              # shoulders
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (5, 11), (6, 12),    # shoulder to hip
    (11, 12)             # hips
]

# Joint names
JOINT_NAMES = [
    'nose', 'left_eye', 'right_eye',
    'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip'
]

# Color scheme (soft colors for infants)
COLORS = {
    'head': (255, 200, 200),      # light pink
    'body': (200, 220, 255),      # light blue
    'left_arm': (200, 255, 200),  # light green
    'right_arm': (255, 255, 200), # light yellow
    'skeleton': (100, 150, 255)   # soft blue
}


def draw_keypoints(image, keypoints, confidence=None, threshold=0.3, 
                   radius=3, thickness=2):
    """
    Draw keypoints and skeleton on image
    
    Args:
        image: (H, W, 3) image (BGR format for OpenCV)
        keypoints: (K, 2) keypoint coordinates
        confidence: (K, 1) confidence scores (optional)
        threshold: minimum confidence to draw
        radius: circle radius for keypoints
        thickness: line thickness for skeleton
    """
    vis_image = image.copy()
    num_joints = len(keypoints)
    
    # Draw skeleton connections first
    for connection in SKELETON_CONNECTIONS:
        if connection[0] >= num_joints or connection[1] >= num_joints:
            continue
        
        pt1 = tuple(keypoints[connection[0]].astype(int))
        pt2 = tuple(keypoints[connection[1]].astype(int))
        
        # Check confidence if provided
        if confidence is not None:
            conf1 = confidence[connection[0]][0]
            conf2 = confidence[connection[1]][0]
            if conf1 < threshold or conf2 < threshold:
                continue
        
        # Draw line
        cv2.line(vis_image, pt1, pt2, COLORS['skeleton'], thickness)
    
    # Draw keypoints on top
    for i, (x, y) in enumerate(keypoints):
        # Check confidence
        if confidence is not None and confidence[i][0] < threshold:
            continue
        
        # Determine color based on joint type
        if i <= 4:  # head joints
            color = COLORS['head']
        elif i in [5, 7, 9]:  # left arm
            color = COLORS['left_arm']
        elif i in [6, 8, 10]:  # right arm
            color = COLORS['right_arm']
        else:  # body
            color = COLORS['body']
        
        center = (int(x), int(y))
        
        # Draw filled circle
        cv2.circle(vis_image, center, radius, color, -1)
        # Draw border
        cv2.circle(vis_image, center, radius, (0, 0, 0), 1)
    
    return vis_image


def draw_keypoints_with_labels(image, keypoints, confidence=None, 
                               threshold=0.3, show_names=True):
    """
    Draw keypoints with joint name labels
    """
    vis_image = draw_keypoints(image, keypoints, confidence, threshold)
    
    if show_names:
        for i, (x, y) in enumerate(keypoints):
            if confidence is not None and confidence[i][0] < threshold:
                continue
            
            # Draw label
            text = JOINT_NAMES[i]
            position = (int(x) + 5, int(y) - 5)
            cv2.putText(vis_image, text, position,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return vis_image


def create_comparison_figure(images, keypoints_list, titles=None):
    """
    Create side-by-side comparison of multiple results
    
    Args:
        images: list of images
        keypoints_list: list of keypoint arrays
        titles: list of titles for each subplot
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    
    if n == 1:
        axes = [axes]
    
    for i, (img, kpts) in enumerate(zip(images, keypoints_list)):
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        vis_img = draw_keypoints(img_rgb, kpts)
        
        axes[i].imshow(vis_img)
        axes[i].axis('off')
        if titles:
            axes[i].set_title(titles[i])
    
    plt.tight_layout()
    return fig


def plot_skeleton_3d(keypoints, ax=None):
    """
    Plot skeleton in 3D (mock 3D with depth estimation)
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Use y-coordinate as pseudo depth
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    z = -y * 0.1  # Pseudo depth
    
    # Draw connections
    for connection in SKELETON_CONNECTIONS:
        pts = np.array([
            [x[connection[0]], y[connection[0]], z[connection[0]]],
            [x[connection[1]], y[connection[1]], z[connection[1]]]
        ])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'b-', linewidth=2)
    
    # Draw joints
    ax.scatter(x, y, z, c='r', marker='o', s=50)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')
    ax.set_title('3D Skeleton View')
    
    return ax


def plot_movement_trajectory(keypoints_sequence, joint_indices=None, 
                             joint_names=None):
    """
    Plot movement trajectories over time
    
    Args:
        keypoints_sequence: (T, K, 2) sequence of keypoints
        joint_indices: list of joint indices to plot
        joint_names: names for legend
    """
    if joint_indices is None:
        joint_indices = [0, 5, 6, 9, 10]  # nose, shoulders, wrists
    
    if joint_names is None:
        joint_names = [JOINT_NAMES[i] for i in joint_indices]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    time_steps = np.arange(len(keypoints_sequence))
    
    # Plot X coordinates
    for idx, name in zip(joint_indices, joint_names):
        x_coords = keypoints_sequence[:, idx, 0]
        ax1.plot(time_steps, x_coords, label=name, marker='o', markersize=2)
    
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('X Coordinate')
    ax1.set_title('X-axis Movement')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Y coordinates
    for idx, name in zip(joint_indices, joint_names):
        y_coords = keypoints_sequence[:, idx, 1]
        ax2.plot(time_steps, y_coords, label=name, marker='o', markersize=2)
    
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_title('Y-axis Movement')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_movement_heatmap(keypoints_sequence, image_shape=(480, 640)):
    """
    Create heatmap showing frequency of joint positions
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Select key joints to visualize
    key_joints = [0, 5, 6, 9, 10, 11]  # nose, shoulders, wrists, left hip
    
    for idx, joint_idx in enumerate(key_joints):
        if idx >= len(axes):
            break
        
        # Create 2D histogram
        x_coords = keypoints_sequence[:, joint_idx, 0]
        y_coords = keypoints_sequence[:, joint_idx, 1]
        
        heatmap, xedges, yedges = np.histogram2d(
            x_coords, y_coords,
            bins=[image_shape[1]//10, image_shape[0]//10],
            range=[[0, image_shape[1]], [0, image_shape[0]]]
        )
        
        im = axes[idx].imshow(heatmap.T, origin='lower', 
                             extent=[0, image_shape[1], 0, image_shape[0]],
                             cmap='hot', interpolation='bilinear')
        axes[idx].set_title(f'{JOINT_NAMES[joint_idx]} Movement Heatmap')
        axes[idx].set_xlabel('X')
        axes[idx].set_ylabel('Y')
        plt.colorbar(im, ax=axes[idx])
    
    plt.tight_layout()
    return fig


def plot_confidence_over_time(confidence_sequence, joint_indices=None):
    """
    Plot confidence scores over time
    """
    if joint_indices is None:
        joint_indices = range(confidence_sequence.shape[1])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    time_steps = np.arange(len(confidence_sequence))
    
    for idx in joint_indices:
        conf = confidence_sequence[:, idx, 0]
        ax.plot(time_steps, conf, label=JOINT_NAMES[idx], alpha=0.7)
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Confidence Score')
    ax.set_title('Detection Confidence Over Time')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig


def create_video_with_pose(video_path, keypoints_sequence, output_path, 
                           show_trajectory=True, trail_length=10):
    """
    Create video with pose overlay and optional trajectory trails
    
    Args:
        video_path: input video path
        keypoints_sequence: (T, K, 2) keypoint predictions
        output_path: output video path
        show_trajectory: whether to show movement trails
        trail_length: number of frames to show in trail
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx < len(keypoints_sequence):
            keypoints = keypoints_sequence[frame_idx]
            
            # Draw current pose
            frame = draw_keypoints(frame, keypoints)
            
            # Draw trajectory trails
            if show_trajectory and frame_idx > 0:
                start_idx = max(0, frame_idx - trail_length)
                for t in range(start_idx, frame_idx):
                    alpha = (t - start_idx) / trail_length
                    prev_kpts = keypoints_sequence[t]
                    
                    # Draw fading trail for key joints (wrists)
                    for joint_idx in [9, 10]:  # left and right wrist
                        pt = tuple(prev_kpts[joint_idx].astype(int))
                        color = (255, 255, 0)  # yellow trail
                        radius = max(1, int(3 * alpha))
                        cv2.circle(frame, pt, radius, color, -1)
            
            # Add frame number
            cv2.putText(frame, f'Frame: {frame_idx}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    print(f"Video with pose saved to {output_path}")


def save_visualization_grid(images, keypoints_list, output_path, 
                            grid_size=(3, 3)):
    """
    Save a grid of visualization results
    """
    rows, cols = grid_size
    n_images = min(len(images), rows * cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten()
    
    for i in range(n_images):
        img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        vis_img = draw_keypoints(img_rgb, keypoints_list[i])
        
        axes[i].imshow(vis_img)
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}')
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization grid saved to {output_path}")


# Clinical analysis visualizations

def plot_movement_amplitude(keypoints_sequence, output_path=None):
    """
    Plot movement amplitude for clinical assessment
    """
    from utils.metrics import calculate_movement_amplitude
    
    amplitude = calculate_movement_amplitude(keypoints_sequence)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    joints = np.arange(len(JOINT_NAMES))
    ax.bar(joints, amplitude, color='skyblue', edgecolor='navy')
    ax.set_xticks(joints)
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha='right')
    ax.set_ylabel('Movement Amplitude (pixels)')
    ax.set_title('Joint Movement Amplitude Analysis')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_clinical_report_figure(keypoints_sequence, confidence_sequence):
    """
    Create comprehensive clinical report visualization
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Movement trajectories
    ax1 = fig.add_subplot(gs[0, :])
    time_steps = np.arange(len(keypoints_sequence))
    for joint_idx in [5, 6, 9, 10]:  # shoulders and wrists
        trajectory = np.sqrt(
            (keypoints_sequence[:, joint_idx, 0] - keypoints_sequence[0, joint_idx, 0])**2 +
            (keypoints_sequence[:, joint_idx, 1] - keypoints_sequence[0, joint_idx, 1])**2
        )
        ax1.plot(time_steps, trajectory, label=JOINT_NAMES[joint_idx])
    ax1.set_xlabel('Time (frames)')
    ax1.set_ylabel('Distance from Origin (pixels)')
    ax1.set_title('Movement Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Confidence analysis
    ax2 = fig.add_subplot(gs[1, 0])
    mean_conf = confidence_sequence.mean(axis=0).flatten()
    ax2.bar(range(len(JOINT_NAMES)), mean_conf, color='lightcoral')
    ax2.set_xticks(range(len(JOINT_NAMES)))
    ax2.set_xticklabels(JOINT_NAMES, rotation=45, ha='right')
    ax2.set_ylabel('Average Confidence')
    ax2.set_title('Detection Quality per Joint')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Movement amplitude
    ax3 = fig.add_subplot(gs[1, 1])
    from utils.metrics import calculate_movement_amplitude
    amplitude = calculate_movement_amplitude(keypoints_sequence)
    ax3.bar(range(len(JOINT_NAMES)), amplitude, color='lightgreen')
    ax3.set_xticks(range(len(JOINT_NAMES)))
    ax3.set_xticklabels(JOINT_NAMES, rotation=45, ha='right')
    ax3.set_ylabel('Amplitude (pixels)')
    ax3.set_title('Range of Motion')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Temporal consistency
    ax4 = fig.add_subplot(gs[2, :])
    velocities = np.diff(keypoints_sequence, axis=0)
    velocity_magnitude = np.sqrt((velocities**2).sum(axis=2))
    for joint_idx in [5, 6, 9, 10]:
        ax4.plot(velocity_magnitude[:, joint_idx], label=JOINT_NAMES[joint_idx])
    ax4.set_xlabel('Time (frames)')
    ax4.set_ylabel('Velocity (pixels/frame)')
    ax4.set_title('Movement Velocity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle('Clinical Movement Analysis Report', fontsize=16, fontweight='bold')
    
    return fig
