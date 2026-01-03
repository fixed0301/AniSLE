"""
ControlNet + Stable Diffusion Inpainting Demo
Uses ControlNet for pose conditioning with SD inpainting for local edits
"""

import os
import sys
import torch
from PIL import Image, ImageDraw
import numpy as np
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)

from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, StableDiffusionControlNetInpaintPipeline
from diffusers import DDIMScheduler
import json
import cv2


def keypoints_to_pose_image(keypoints, image_size=(512, 512)):
    """
    Convert keypoints JSON to ControlNet-compatible pose image
    Uses OpenPose visualization format that ControlNet expects
    """
    import numpy as np
    
    # Check if keypoints need scaling
    max_coord = max(max(kp[0], kp[1]) for kp in keypoints)
    target_size = image_size[0]
    
    if max_coord <= 256:
        scale = target_size / 256
        scaled_keypoints = [[kp[0] * scale, kp[1] * scale] for kp in keypoints]
        print(f"  Scaled keypoints from 256 to {target_size}")
    else:
        scaled_keypoints = [[kp[0], kp[1]] for kp in keypoints]
        print(f"  Keypoints already at {target_size}")
    
    # Ensure 18 keypoints
    while len(scaled_keypoints) < 18:
        scaled_keypoints.append([0, 0])
    
    # Create canvas
    canvas = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    
    # OpenPose connections with colors (ControlNet format)
    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8],  # Arms
        [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14],  # Legs
        [2, 1], [1, 15], [15, 17], [1, 16], [16, 18],  # Head
        [3, 17], [6, 18]  # Shoulders to ears
    ]
    
    # Colors for each limb (OpenPose standard)
    colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], 
        [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
        [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], 
        [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], 
        [255, 0, 170], [255, 0, 85], [255, 0, 0]
    ]
    
    # Draw limbs
    for i, limb in enumerate(limbSeq):
        idx1, idx2 = limb[0] - 1, limb[1] - 1  # Convert to 0-indexed
        if idx1 < len(scaled_keypoints) and idx2 < len(scaled_keypoints):
            Y = [scaled_keypoints[idx1][0], scaled_keypoints[idx2][0]]
            X = [scaled_keypoints[idx1][1], scaled_keypoints[idx2][1]]
            
            if Y[0] > 0 and Y[1] > 0:  # Valid keypoints
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = np.degrees(np.arctan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(canvas, polygon, colors[i % len(colors)])
    
    # Draw keypoints
    for i, point in enumerate(scaled_keypoints[:18]):
        x, y = int(point[0]), int(point[1])
        if x > 0 and y > 0:
            cv2.circle(canvas, (x, y), 4, (255, 255, 255), thickness=-1)
    
    return Image.fromarray(canvas)


def run_controlnet_inpainting(
    source_image_path,
    target_pose_json,
    mask_path,
    target_pose_image=None,
    output_path="data/output.png",
    prompt="person's arms, human arms",
    negative_prompt="blurry, distorted, low quality, duplicate limbs, extra arms, deformed",
    num_steps=50,
    guidance_scale=12.0,
    controlnet_scale=1.5,
    strength=0.99,
    expand_mask=False,
    seed=42,
    device="cuda"
):
    """
    Run ControlNet + SD Inpainting
    
    Args:
        source_image_path: Source image
        target_pose_json: JSON file with target pose keypoints
        mask_path: Inpainting mask (white=inpaint, black=keep)
        target_pose_image: Optional - extract pose from this image instead of JSON
        output_path: Output path
        strength: Inpainting strength (higher = more change)
        expand_mask: Expand mask to include both source and target arm regions
    """
    
    print(f"\n{'='*60}")
    print("ControlNet + Stable Diffusion Inpainting")
    print(f"{'='*60}")
    print(f"Source: {source_image_path}")
    if target_pose_image:
        print(f"Target pose: extracted from {target_pose_image}")
    else:
        print(f"Target pose JSON: {target_pose_json}")
    print(f"Mask: {mask_path}")
    print(f"Prompt: {prompt}")
    print(f"Steps: {num_steps}, Guidance: {guidance_scale}, ControlNet: {controlnet_scale}, Strength: {strength}")
    
    image_size = (512, 512)
    
    # Load source image
    source_img = Image.open(source_image_path).convert("RGB").resize(image_size, Image.BICUBIC)
    
    # Load mask
    mask_img = Image.open(mask_path).convert("L").resize(image_size, Image.BICUBIC)
    mask_array = np.array(mask_img)
    
    # Expand mask to include source arm regions if needed
    if expand_mask:
        print("\nExpanding mask to include source pose regions...")
        # Extract source pose
        from single_extract_pose import inference_pose
        source_pose = inference_pose(source_image_path, image_size=image_size)
        
        # Create mask from source pose (where pose skeleton exists)
        source_pose_array = np.array(source_pose)
        source_pose_mask = (source_pose_array.sum(axis=2) > 0).astype(np.uint8) * 255
        
        # Dilate source pose mask to cover arm regions
        import cv2
        kernel = np.ones((30, 30), np.uint8)
        source_pose_mask_dilated = cv2.dilate(source_pose_mask, kernel, iterations=1)
        
        # Combine with original mask (union)
        combined_mask = np.maximum(mask_array, source_pose_mask_dilated)
        mask_img = Image.fromarray(combined_mask)
        
        # Save debug mask
        debug_dir = os.path.dirname(output_path) or '.'
        combined_mask_path = os.path.join(debug_dir, 'debug_expanded_mask.png')
        mask_img.save(combined_mask_path)
        print(f"  Original mask: {(mask_array > 128).sum() / mask_array.size * 100:.1f}%")
        print(f"  Expanded mask: {(combined_mask > 128).sum() / combined_mask.size * 100:.1f}%")
        print(f"  DEBUG: Expanded mask saved to {combined_mask_path}")
    
    # Get target pose
    if target_pose_image:
        # Extract pose from real image using DWPose
        print("\nExtracting pose from target image...")
        from single_extract_pose import inference_pose
        pose_image = inference_pose(target_pose_image, image_size=image_size)
        print(f"  Pose extracted successfully")
    else:
        # Load and convert JSON keypoints to ControlNet format
        print("\nLoading target pose from JSON...")
        with open(target_pose_json, 'r') as f:
            keypoints = json.load(f)
        print(f"  Loaded {len(keypoints)} keypoints")
        
        # Convert to ControlNet-compatible pose image
        pose_image = keypoints_to_pose_image(keypoints, image_size)
        print(f"  Converted to ControlNet pose format")
    
    # Save debug pose
    debug_dir = os.path.dirname(output_path) or '.'
    os.makedirs(debug_dir, exist_ok=True)
    pose_debug_path = os.path.join(debug_dir, 'debug_controlnet_pose.png')
    pose_image.save(pose_debug_path)
    print(f"  DEBUG: Pose skeleton saved to {pose_debug_path}")
    
    # Load ControlNet model for pose
    print("\nLoading ControlNet (OpenPose)...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose",
        torch_dtype=torch.float16
    ).to(device)
    
    # Load SD Inpainting pipeline with ControlNet
    print("Loading Stable Diffusion Inpainting + ControlNet pipeline...")
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    print("Pipeline loaded successfully!")
    
    # Run inference
    print(f"\nGenerating with ControlNet conditioning...")
    print(f"  Using strength={strength} for aggressive inpainting")
    generator = torch.Generator(device=device).manual_seed(seed)
    
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=source_img,
        mask_image=mask_img,
        control_image=pose_image,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_scale,
        strength=strength,
        generator=generator,
        height=image_size[1],
        width=image_size[0]
    ).images[0]
    
    # Save result
    result.save(output_path)
    print(f"\n✓ Saved to: {output_path}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ControlNet + SD Inpainting Demo")
    parser.add_argument('--idx', required=True, help='Image index (source)')
    parser.add_argument('--target-idx', default=None, help='Target image index for pose (default: use aligned pose of source)')
    parser.add_argument('--prompt', default='person arms, human arms', help='Text prompt')
    parser.add_argument('--steps', type=int, default=50, help='Inference steps')
    parser.add_argument('--guidance', type=float, default=12.0, help='Guidance scale')
    parser.add_argument('--controlnet-scale', type=float, default=1.5, help='ControlNet conditioning scale')
    parser.add_argument('--strength', type=float, default=0.99, help='Inpainting strength (0-1, higher=more change)')
    parser.add_argument('--expand-mask', action='store_true', default=False, help='Expand mask to include source pose')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', default='cuda', help='Device')
    
    args = parser.parse_args()
    idx = args.idx
    
    # Build paths
    upload_dir = '../../flask_app/data/uploads'
    source_path = None
    for ext in ['.png', '.jpeg', '.jpg']:
        test_path = f'{upload_dir}/{idx}{ext}'
        if os.path.exists(test_path):
            source_path = test_path
            break
    
    if source_path is None:
        raise FileNotFoundError(f"Source not found: {upload_dir}/{idx}.*")
    
    pose_json_path = f'../pose/pose_json/keypoints_aligned_{idx}.json'
    mask_path = f'../mask/mask_img/mask_{idx}.png'
    output_path = f'../../flask_app/data/results/{idx}_controlnet.png'
    
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    
    # Determine pose source
    if args.target_idx:
        # Use pose from different image
        target_image_path = None
        for ext in ['.png', '.jpeg', '.jpg']:
            test_path = f'{upload_dir}/{args.target_idx}{ext}'
            if os.path.exists(test_path):
                target_image_path = test_path
                break
        
        if target_image_path is None:
            raise FileNotFoundError(f"Target image not found: {upload_dir}/{args.target_idx}.*")
        
        print(f"\n>>> Using pose extracted from: {target_image_path}")
        pose_json_path = None  # Will extract from image
    else:
        # Use aligned JSON pose
        pose_json_path = f'../pose/pose_json/keypoints_aligned_{idx}.json'
        target_image_path = None
        print(f"\n>>> Using aligned pose JSON: {pose_json_path}")
    
    run_controlnet_inpainting(
        source_image_path=source_path,
        target_pose_json=pose_json_path,
        target_pose_image=target_image_path,
        mask_path=mask_path,
        output_path=output_path,
        prompt=args.prompt,
        num_steps=args.steps,
        guidance_scale=args.guidance,
        controlnet_scale=args.controlnet_scale,
        strength=args.strength,
        expand_mask=args.expand_mask,
        seed=args.seed,
        device=args.device
    )
