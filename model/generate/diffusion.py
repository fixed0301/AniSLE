"""
IP-Adapter + ControlNet + SD Inpainting
Preserves appearance while changing pose in masked region
"""

import os
import sys
import torch
from PIL import Image
import numpy as np
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)

from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from diffusers import DDIMScheduler
from diffusers.utils import load_image
import json


def keypoints_to_pose_image(keypoints, image_size=(512, 512)):
    """Convert keypoints to ControlNet-compatible OpenPose image"""
    
    max_coord = max(max(kp[0], kp[1]) for kp in keypoints)
    target_size = image_size[0]
    
    if max_coord <= 256:
        scale = target_size / 256
        scaled_keypoints = [[kp[0] * scale, kp[1] * scale] for kp in keypoints]
        print(f"  Scaled keypoints from 256 to {target_size}")
    else:
        scaled_keypoints = [[kp[0], kp[1]] for kp in keypoints]
        print(f"  Keypoints already at {target_size}")
    
    while len(scaled_keypoints) < 18:
        scaled_keypoints.append([0, 0])
    
    canvas = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    
    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8],
        [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14],
        [2, 1], [1, 15], [15, 17], [1, 16], [16, 18],
        [3, 17], [6, 18]
    ]
    
    colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
        [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
        [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
        [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
        [255, 0, 170], [255, 0, 85], [255, 0, 0]
    ]
    
    for i, limb in enumerate(limbSeq):
        idx1, idx2 = limb[0] - 1, limb[1] - 1
        if idx1 < len(scaled_keypoints) and idx2 < len(scaled_keypoints):
            # keypoints are [x, y] format
            X = [scaled_keypoints[idx1][0], scaled_keypoints[idx2][0]]
            Y = [scaled_keypoints[idx1][1], scaled_keypoints[idx2][1]]
            
            if X[0] > 0 and X[1] > 0:
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = np.degrees(np.arctan2(Y[0] - Y[1], X[0] - X[1]))
                polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), 4), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(canvas, polygon, colors[i % len(colors)])
    
    for i, point in enumerate(scaled_keypoints[:18]):
        x, y = int(point[0]), int(point[1])
        if x > 0 and y > 0:
            cv2.circle(canvas, (x, y), 4, (255, 255, 255), thickness=-1)
    
    return Image.fromarray(canvas)


def run_ipadapter_controlnet(
    source_image_path,
    target_pose_json,
    mask_path,
    output_path="data/output.png",
    prompt="detailed human arm, natural pose, realistic skin texture, proper arm anatomy, high quality",
    negative_prompt="blurry, distorted, deformed, ugly, bad anatomy, extra limbs, multiple arms, disconnected limbs, low quality",
    num_steps=50,
    guidance_scale=9.0,
    controlnet_scale=5.0,
    ip_adapter_scale=0.5,
    strength=0.85,
    expand_arm_mask=False,
    seed=42,
    device="cuda"
):
    """
    Run IP-Adapter + ControlNet + SD Inpainting
    
    Args:
        source_image_path: Source image (for appearance)
        target_pose_json: Target pose keypoints
        mask_path: Inpainting mask
        ip_adapter_scale: IP-Adapter strength (0-1, higher=more appearance preservation)
        expand_arm_mask: Expand mask to include source arm regions
    """
    
    print(f"\n{'='*60}")
    print("IP-Adapter + ControlNet + SD Inpainting")
    print(f"{'='*60}")
    print(f"Source: {source_image_path}")
    print(f"Target pose: {target_pose_json}")
    print(f"Mask: {mask_path}")
    print(f"IP-Adapter scale: {ip_adapter_scale} (appearance preservation)")
    print(f"ControlNet scale: {controlnet_scale} (pose control)")
    print(f"Steps: {num_steps}, Guidance: {guidance_scale}, Strength: {strength}")
    
    image_size = (512, 512)
    
    # Load source image
    source_img = Image.open(source_image_path).convert("RGB").resize(image_size, Image.BICUBIC)
    
    # Load mask
    mask_img = Image.open(mask_path).convert("L").resize(image_size, Image.BICUBIC)
    mask_array = np.array(mask_img)
    
    # Expand mask to include source arm regions
    if expand_arm_mask:
        print("\nExpanding mask to include source arm regions...")
        from single_extract_pose import inference_pose
        
        # Extract source pose
        source_pose_img = inference_pose(source_image_path, image_size=image_size)
        source_pose_array = np.array(source_pose_img)
        
        # Create mask only from arm regions (not whole body)
        # Detect colored pixels in pose (arms are usually colored)
        arm_mask = (source_pose_array.sum(axis=2) > 50).astype(np.uint8) * 255
        
        # Dilate to cover arm width
        kernel = np.ones((20, 20), np.uint8)
        arm_mask_dilated = cv2.dilate(arm_mask, kernel, iterations=1)
        
        # But restrict to upper body region (y < 350) to avoid legs
        height, width = arm_mask_dilated.shape
        arm_mask_dilated[int(height * 0.7):, :] = 0  # Zero out lower 30%
        
        # Combine with original mask
        combined_mask = np.maximum(mask_array, arm_mask_dilated)
        mask_img = Image.fromarray(combined_mask)
        
        # Save debug
        debug_dir = os.path.dirname(output_path) or '.'
        arm_mask_path = os.path.join(debug_dir, 'debug_arm_mask.png')
        Image.fromarray(arm_mask_dilated).save(arm_mask_path)
        combined_mask_path = os.path.join(debug_dir, 'debug_combined_mask.png')
        mask_img.save(combined_mask_path)
        
        print(f"  Original mask: {(mask_array > 128).sum() / mask_array.size * 100:.1f}%")
        print(f"  + Arm regions: {(arm_mask_dilated > 128).sum() / arm_mask_dilated.size * 100:.1f}%")
        print(f"  = Combined: {(combined_mask > 128).sum() / combined_mask.size * 100:.1f}%")
        print(f"  DEBUG: Arm mask -> {arm_mask_path}")
        print(f"  DEBUG: Combined mask -> {combined_mask_path}")
    
    # Load target pose
    print("\nLoading target pose from JSON...")
    with open(target_pose_json, 'r') as f:
        keypoints_data = json.load(f)
    
    # Extract keypoints array from JSON structure
    if isinstance(keypoints_data, dict) and 'keypoints' in keypoints_data:
        keypoints = keypoints_data['keypoints']
    else:
        keypoints = keypoints_data
    
    print(f"  Loaded {len(keypoints)} keypoints")
    
    pose_image = keypoints_to_pose_image(keypoints, image_size)
    print(f"  Converted to ControlNet pose format")
    
    # Save debug
    debug_dir = os.path.dirname(output_path) or '.'
    os.makedirs(debug_dir, exist_ok=True)
    pose_debug_path = os.path.join(debug_dir, 'debug_ipadapter_pose.png')
    pose_image.save(pose_debug_path)
    print(f"  DEBUG: Pose saved to {pose_debug_path}")
    
    # Load ControlNet
    print("\nLoading ControlNet (OpenPose)...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose",
        torch_dtype=torch.float16
    ).to(device)
    
    # Load pipeline
    print("Loading SD Inpainting + ControlNet pipeline...")
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    # Load IP-Adapter
    print(f"Loading IP-Adapter (scale={ip_adapter_scale})...")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    pipe.set_ip_adapter_scale(ip_adapter_scale)
    
    print("Pipeline loaded successfully!")
    
    # Generate
    print(f"\nGenerating with IP-Adapter + ControlNet...")
    print(f"  IP-Adapter: preserving appearance from source")
    print(f"  ControlNet: following target pose")
    print(f"  Inpainting: only in masked region")
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=source_img,
        mask_image=mask_img,
        control_image=pose_image,
        ip_adapter_image=source_img,  # Use source for appearance
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_scale,
        strength=strength,
        generator=generator,
        height=image_size[1],
        width=image_size[0]
    ).images[0]
    
    # Save
    result.save(output_path)
    print(f"\n✓ Saved to: {output_path}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IP-Adapter + ControlNet Inpainting")
    parser.add_argument('--idx', required=True, help='Image index')
    parser.add_argument('--prompt', default='', help='Optional text prompt')
    parser.add_argument('--steps', type=int, default=50, help='Inference steps')
    parser.add_argument('--guidance', type=float, default=7.5, help='Guidance scale')
    parser.add_argument('--controlnet-scale', type=float, default=2.0, help='ControlNet scale')
    parser.add_argument('--ip-scale', type=float, default=0.8, help='IP-Adapter scale (0-1)')
    parser.add_argument('--strength', type=float, default=0.85, help='Inpainting strength')
    parser.add_argument('--expand-arm', action='store_true', default=False, help='Auto-expand mask to cover source arms')
    parser.add_argument('--no-expand-arm', dest='expand_arm', action='store_false', help='Disable auto-expand mask')
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
    output_path = f'../../flask_app/data/results/{idx}_ipadapter.png'
    
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    
    run_ipadapter_controlnet(
        source_image_path=source_path,
        target_pose_json=pose_json_path,
        mask_path=mask_path,
        output_path=output_path,
        prompt=args.prompt,
        num_steps=args.steps,
        guidance_scale=args.guidance,
        controlnet_scale=args.controlnet_scale,
        ip_adapter_scale=args.ip_scale,
        strength=args.strength,
        expand_arm_mask=args.expand_arm,
        seed=args.seed,
        device=args.device
    )
