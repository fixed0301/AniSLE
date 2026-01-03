"""
Generate mask based on geometric difference between source and target keypoints
"""

import json
import numpy as np
from PIL import Image, ImageDraw
import os

# OpenPose body connections (18 keypoints)
BODY_CONNECTIONS = [
    (1, 2), (1, 5),  # Neck to shoulders
    (2, 3), (3, 4),  # Right arm
    (5, 6), (6, 7),  # Left arm
    (1, 8),  # Neck to mid hip
    (8, 9), (9, 10),  # Right hip and leg
    (8, 11), (11, 12), (12, 13),  # Left hip and leg
    (1, 0),  # Neck to nose
    (0, 14), (0, 15),  # Nose to eyes
    (14, 16), (15, 17),  # Eyes to ears
]

def load_keypoints(json_path, image_size=256):
    """Load keypoints from JSON file and normalize to pixel coordinates"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON formats
    if isinstance(data, list):
        # Simple list format: [[x, y], [x, y], ...] - already in pixel coords
        keypoints = np.array(data[:18])
        print(f"  Loaded list format: {len(keypoints)} keypoints")
    elif isinstance(data, dict) and 'bodies' in data:
        # Full format with bodies/hands/faces
        bodies = np.array(data['bodies'])[:18]
        print(f"  Loaded dict format: {len(bodies)} keypoints, range: [{bodies.min():.3f}, {bodies.max():.3f}]")
        
        # Check if normalized (0-1 range, with small margin) or pixel coordinates
        if bodies.max() <= 1.1 and bodies.min() >= 0:
            # Normalized coordinates - convert to pixels
            keypoints = bodies * image_size
            print(f"  Converted normalized coords to pixels (scale: {image_size})")
        else:
            keypoints = bodies
            print(f"  Using pixel coordinates as-is")
    else:
        raise ValueError(f"Unsupported JSON format in {json_path}")
    
    print(f"  Final keypoints range: [{keypoints.min():.1f}, {keypoints.max():.1f}]")
    return keypoints

def calculate_keypoint_distances(kp_source, kp_target):
    """Calculate L2 distance between corresponding keypoints"""
    # Ensure same number of keypoints
    min_len = min(len(kp_source), len(kp_target))
    kp_source = kp_source[:min_len]
    kp_target = kp_target[:min_len]
    
    # Calculate L2 distances
    distances = np.linalg.norm(kp_source - kp_target, axis=1)
    return distances

def generate_mask_from_keypoint_diff(
    source_kp_path,
    target_kp_path,
    output_path,
    image_size=(256, 256),
    threshold=10.0,
    joint_radius=15,
    limb_thickness=10,
    scale_factor=1.0,
    body_parts=None  # e.g., ['arms', 'legs', 'upper_body']
):
    """
    Generate mask based on keypoint differences
    
    Args:
        source_kp_path: Path to source keypoints JSON
        target_kp_path: Path to target keypoints JSON
        output_path: Path to save mask image
        image_size: Output mask size (width, height)
        threshold: Minimum distance to consider as "changed" (in pixels)
        joint_radius: Radius to draw around moved joints
        limb_thickness: Thickness of limbs connecting moved joints
        scale_factor: Scale factor if keypoints are from different image size
        body_parts: List of body parts to include ['arms', 'legs', 'upper_body', 'lower_body', 'head']
    """
    
    # Define body part groups (joint indices)
    BODY_PART_GROUPS = {
        'head': [0, 14, 15, 16, 17],  # Nose, eyes, ears
        'upper_body': [1, 2, 5],  # Neck, shoulders
        'right_arm': [2, 3, 4],
        'left_arm': [5, 6, 7],
        'arms': [2, 3, 4, 5, 6, 7],  # Both arms
        'torso': [1, 8, 11],
        'right_leg': [8, 9, 10],
        'left_leg': [11, 12, 13],
        'legs': [8, 9, 10, 11, 12, 13],  # Both legs
        'lower_body': [8, 9, 10, 11, 12, 13],
    }
    
    # Define limb connections by body part
    LIMB_GROUPS = {
        'head': [(1, 0), (0, 14), (0, 15), (14, 16), (15, 17)],
        'upper_body': [(1, 2), (1, 5)],
        'right_arm': [(2, 3), (3, 4)],
        'left_arm': [(5, 6), (6, 7)],
        'arms': [(2, 3), (3, 4), (5, 6), (6, 7)],
        'torso': [(1, 8)],
        'right_leg': [(8, 9), (9, 10)],
        'left_leg': [(8, 11), (11, 12), (12, 13)],
        'legs': [(8, 9), (9, 10), (8, 11), (11, 12), (12, 13)],
        'lower_body': [(8, 9), (9, 10), (8, 11), (11, 12), (12, 13)],
    }
    
    print(f"Loading keypoints...")
    print(f"  Source: {source_kp_path}")
    print(f"  Target: {target_kp_path}")
    
    # Load keypoints (pass image size for proper normalization)
    kp_source = load_keypoints(source_kp_path, image_size=image_size[0])
    kp_target = load_keypoints(target_kp_path, image_size=image_size[0])
    
    # Scale keypoints if needed
    if scale_factor != 1.0:
        kp_source = kp_source * scale_factor
        kp_target = kp_target * scale_factor
    
    print(f"  Source keypoints: {kp_source.shape}")
    print(f"  Target keypoints: {kp_target.shape}")
    
    # Calculate distances
    distances = calculate_keypoint_distances(kp_source, kp_target)
    
    # Find which keypoints moved significantly
    moved_joints = distances > threshold
    moved_indices = np.where(moved_joints)[0]
    
    # Filter by body parts if specified
    if body_parts:
        allowed_joints = set()
        allowed_limbs = []
        
        for part in body_parts:
            if part in BODY_PART_GROUPS:
                allowed_joints.update(BODY_PART_GROUPS[part])
            if part in LIMB_GROUPS:
                allowed_limbs.extend(LIMB_GROUPS[part])
        
        # Filter moved joints by body parts
        moved_indices = [idx for idx in moved_indices if idx in allowed_joints]
        limbs_to_draw = allowed_limbs
        print(f"  Filtering by body parts: {body_parts}")
        print(f"  Allowed joints: {sorted(allowed_joints)}")
    else:
        limbs_to_draw = BODY_CONNECTIONS
    
    print(f"\nAnalyzing movement:")
    print(f"  Threshold: {threshold} pixels")
    print(f"  Moved joints: {len(moved_indices)}/{len(distances)}")
    
    for idx in moved_indices:
        print(f"    Joint {idx}: moved {distances[idx]:.1f} pixels")
    
    # Create mask (black background)
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    
    # Draw limbs that have at least one moved joint
    print(f"\nDrawing mask regions...")
    limbs_drawn = 0
    
    moved_joints_set = set(moved_indices)
    for start_idx, end_idx in limbs_to_draw:
        if start_idx >= len(distances) or end_idx >= len(distances):
            continue
        
        # If either joint moved significantly, draw the limb
        if start_idx in moved_joints_set or end_idx in moved_joints_set:
            # Draw BOTH source and target positions
            # Source position (old pose - to be erased)
            start_point_src = tuple(kp_source[start_idx].astype(int))
            end_point_src = tuple(kp_source[end_idx].astype(int))
            draw.line([start_point_src, end_point_src], fill=255, width=limb_thickness)
            
            # Target position (new pose - to be generated)
            start_point_tgt = tuple(kp_target[start_idx].astype(int))
            end_point_tgt = tuple(kp_target[end_idx].astype(int))
            draw.line([start_point_tgt, end_point_tgt], fill=255, width=limb_thickness)
            limbs_drawn += 1
    
    print(f"  Drew {limbs_drawn} limbs (both source and target positions)")
    
    # Draw circles around moved joints (BOTH source and target)
    joints_drawn = 0
    for idx in moved_indices:
        if idx < len(kp_target):
            # Source position (old)
            center_src = kp_source[idx].astype(int)
            x_src, y_src = center_src[0], center_src[1]
            draw.ellipse(
                [(x_src - joint_radius, y_src - joint_radius),
                 (x_src + joint_radius, y_src + joint_radius)],
                fill=255
            )
            
            # Target position (new)
            center_tgt = kp_target[idx].astype(int)
            x_tgt, y_tgt = center_tgt[0], center_tgt[1]
            draw.ellipse(
                [(x_tgt - joint_radius, y_tgt - joint_radius),
                 (x_tgt + joint_radius, y_tgt + joint_radius)],
                fill=255
            )
            joints_drawn += 1
    
    print(f"  Drew {joints_drawn} joint circles (both source and target positions)")
    
    # Calculate mask coverage
    mask_array = np.array(mask)
    coverage = (mask_array > 0).sum() / mask_array.size * 100
    
    print(f"\nMask statistics:")
    print(f"  Coverage: {coverage:.1f}%")
    print(f"  White pixels: {(mask_array > 0).sum()}")
    
    # Save mask
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    mask.save(output_path)
    print(f"\n✓ Mask saved to: {output_path}")
    
    return mask, moved_indices, distances


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate mask from keypoint differences")
    parser.add_argument('--source', default='../pose/pose_json/keypoints_0.json',
                       help='Source keypoints JSON path')
    parser.add_argument('--target', default='../pose/pose_json/keypoints_aligned_0.json',
                       help='Target keypoints JSON path')
    parser.add_argument('--output', default='mask_img/mask_keypoint_diff.png',
                       help='Output mask image path')
    parser.add_argument('--size', type=int, nargs=2, default=[256, 256],
                       help='Output image size (width height)')
    parser.add_argument('--threshold', type=float, default=10.0,
                       help='Movement threshold in pixels')
    parser.add_argument('--joint-radius', type=int, default=25,
                       help='Radius around moved joints')
    parser.add_argument('--limb-thickness', type=int, default=30,
                       help='Thickness of limbs')
    parser.add_argument('--scale', type=float, default=1.0,
                       help='Scale factor for keypoints')
    parser.add_argument('--body-parts', nargs='+',
                       choices=['head', 'upper_body', 'arms', 'right_arm', 'left_arm',
                               'torso', 'legs', 'right_leg', 'left_leg', 'lower_body'],
                       help='Body parts to include in mask')
    
    args = parser.parse_args()
    
    generate_mask_from_keypoint_diff(
        source_kp_path=args.source,
        target_kp_path=args.target,
        output_path=args.output,
        image_size=tuple(args.size),
        threshold=args.threshold,
        joint_radius=args.joint_radius,
        limb_thickness=args.limb_thickness,
        scale_factor=args.scale,
        body_parts=args.body_parts
    )
