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

from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, AutoencoderKL
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import load_image
import json


def keypoints_to_pose_image(keypoints, image_size=(512, 512)):
    w, h = image_size[0], image_size[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
        [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
        [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
        [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
        [255, 0, 170], [255, 0, 85]
    ]

    limbs = [
        [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
        [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
        [1, 16], [16, 18]
    ]

    for i, p in enumerate(keypoints[:18]):
        if p[0] > 0 and p[1] > 0:
            color = colors[i]
            cv2.circle(canvas, (int(p[0]), int(p[1])), 8, color, -1)

    for i, limb in enumerate(limbs):
        idx1, idx2 = limb[0] - 1, limb[1] - 1
        if idx1 < len(keypoints) and idx2 < len(keypoints):
            p1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
            p2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
            if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                color = colors[i % len(colors)]
                cv2.line(canvas, p1, p2, color, thickness=6, lineType=cv2.LINE_AA)

    return Image.fromarray(canvas)


def run_ipadapter_controlnet(
    source_image_path,
    target_pose_json,
    mask_path,
    output_path="data/output.png",
    prompt="masterpiece, best quality, detailed hands, beautiful fingers, anatomically correct, matching reference clothing, consistent fabric texture and color",
    negative_prompt="cropped arms, missing hands, missing fingers, low quality",
    num_steps=30,
    guidance_scale=4.3,
    controlnet_scale=1.2,
    depth_scale=0.8,
    ip_adapter_scale=0.4,
    strength=1.0,
    expand_arm_mask=False,
    skip_mask_processing=False,
    seed=42,
    device="cuda"
):
    """
    Run IP-Adapter + Multi-ControlNet (OpenPose + Depth) + SD Inpainting
    
    Args:
        source_image_path: Source image (for appearance)
        target_pose_json: Target pose keypoints
        mask_path: Inpainting mask
        skip_mask_processing: If True, use mask as-is without blur/dilation/feathering
        controlnet_scale: OpenPose ControlNet strength (pose skeleton)
        depth_scale: Depth ControlNet strength (arm volume and perspective)
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
    
    image_size = (1024, 1024)  # SDXL native resolution
    
    # Load source image
    source_img = Image.open(source_image_path).convert("RGB").resize(image_size, Image.BICUBIC)
    
    # Load mask with simple blur only (no dilation/erosion)
    mask_img = Image.open(mask_path).convert("L").resize(image_size, Image.BICUBIC)
    mask_array = np.array(mask_img)
    
    # Save original mask for debug
    debug_dir = os.path.dirname(output_path) or '.'
    os.makedirs(debug_dir, exist_ok=True)
    Image.fromarray(mask_array).save(os.path.join(debug_dir, 'debug_mask_original.png'))
    
    if skip_mask_processing:
        # Use mask as-is (already processed by app.py)
        print("\nUsing pre-processed mask (skipping blur/dilation/feathering)")
        mask_img = Image.fromarray(mask_array)
        Image.fromarray(mask_array).save(os.path.join(debug_dir, 'debug_mask_final.png'))
    else:
        # Apply strong Gaussian Blur for very soft edges
        mask_array = cv2.GaussianBlur(mask_array, (71, 71), 0)
        
        # Apply slight dilation to expand mask area
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_array = cv2.dilate(mask_array, kernel, iterations=1)
        Image.fromarray(mask_array).save(os.path.join(debug_dir, 'debug_mask_blurred.png'))
        
        # Create gradient feather at edges only
        mask_normalized = mask_array.astype(np.float32) / 255.0
        
        # Apply distance transform for natural feathering
        mask_binary = (mask_normalized > 0.5).astype(np.uint8)
        dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
        
        # Normalize distance transform to create feathering effect
        if dist_transform.max() > 0:
            # Apply feathering only near edges (within 80 pixels)
            feather_width = 80
            feather_mask = np.clip(dist_transform / feather_width, 0, 1)
            mask_normalized = np.minimum(mask_normalized, feather_mask)
        
        # Final smooth blur
        mask_normalized = cv2.GaussianBlur(mask_normalized, (35, 35), 0)
        mask_array = (mask_normalized * 255).astype(np.uint8)
        
        mask_img = Image.fromarray(mask_array)
        mask_img.save(os.path.join(debug_dir, 'debug_mask_final.png'))
        
        # Also save final processed mask for inspection
        final_mask_path = os.path.join(debug_dir, f'mask_processed_{os.path.basename(mask_path)}')
        mask_img.save(final_mask_path)
        
        print("\nApplied mask smoothing with dilation:")
        print(f"  - Gaussian Blur: 71x71 kernel (strong smoothing)")
        print(f"  - Dilation: 15x15 ellipse kernel (expand mask area)")
        print(f"  - Distance Feathering: 80px gradient at edges")
        print(f"  - Final Blur: 35x35 kernel (polish)")
        print(f"  - Debug masks saved to {debug_dir}/debug_mask_*.png")
        print(f"  - Final processed mask saved to {final_mask_path}")
    
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

    print(f"  Loaded {len(keypoints)} keypoints: {keypoints}")

    # 디버깅: 스케일링 전후 keypoints 좌표 출력
    max_coord = max(max(kp[0], kp[1]) for kp in keypoints)
    image_size_val = image_size[0]
    if max_coord <= 256:
        scale = image_size_val / 256
    elif max_coord <= 512:
        scale = image_size_val / 512
    else:
        scale = 1.0
    scaled_keypoints = [[kp[0] * scale, kp[1] * scale] for kp in keypoints]
   
    # 실제 pose image 생성은 기존 함수 사용
    pose_image = keypoints_to_pose_image(scaled_keypoints, image_size)
    print(f"  Converted to ControlNet pose format (COCO standard colors)")

    # Save debug
    debug_dir = os.path.dirname(output_path) or '.'
    os.makedirs(debug_dir, exist_ok=True)
    pose_debug_path = os.path.join(debug_dir, 'debug_ipadapter_pose.png')
    pose_image.save(pose_debug_path)
    print(f"  DEBUG: Pose saved to {pose_debug_path}")
    
    # Load ControlNet for SDXL (MUST be SDXL-compatible!)
    print(f"\nLoading SDXL ControlNet (OpenPose, scale={controlnet_scale})...")
    controlnet = ControlNetModel.from_pretrained(
        "xinsir/controlnet-openpose-sdxl-1.0",  # SDXL OpenPose (most stable)
        torch_dtype=torch.float16
    ).to(device)
    
    # Load fixed VAE to prevent burnt colors on white backgrounds
    print("Loading fixed SDXL VAE (madebyollin/sdxl-vae-fp16-fix)...")
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16
    ).to(device)
    print("  ✓ Fixed VAE loaded (prevents black spots and burnt boundaries)")
    
    # Load Animagine XL 3.1 pipeline with fixed VAE
    print("Loading Animagine XL 3.1 from local checkpoint...")
    print("  Using Animagine XL 3.1 for high-quality anime generation...")
    
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
        "/workspace/AniSLE/model/checkpoints/animagine-xl-3.1.safetensors",
        controlnet=controlnet,
        vae=vae,  # Use fixed VAE instead of built-in
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(device)
    
    # DPM++ 2M scheduler with Karras for sharp lines
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++"
    )
    
    # Load IP-Adapter for appearance preservation
    print(f"Loading IP-Adapter (scale={ip_adapter_scale})...")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    pipe.set_ip_adapter_scale(ip_adapter_scale)
    
    print("Pipeline loaded successfully!")
    print("  Using Animagine XL 3.1 for high-quality anime generation")
    
    # Generate
    print(f"\nGenerating with SDXL + ControlNet + IP-Adapter...")
    print(f"  IP-Adapter: preserving source appearance (scale={ip_adapter_scale})")
    print(f"  ControlNet: following target pose skeleton (scale={controlnet_scale})")
    print(f"  Inpainting: only in masked region (strength={strength})")
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=source_img,
        mask_image=mask_img,
        control_image=pose_image,
        ip_adapter_image=source_img,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        guidance_rescale=0.7,
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
    parser.add_argument('--controlnet-scale', type=float, default=2.0, help='ControlNet OpenPose scale')
    parser.add_argument('--depth-scale', type=float, default=0.8, help='ControlNet Depth scale (for arm volume)')
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
    
    # Build kwargs conditionally based on prompt
    kwargs = {
        'source_image_path': source_path,
        'target_pose_json': pose_json_path,
        'mask_path': mask_path,
        'output_path': output_path,
        'num_steps': args.steps,
        'guidance_scale': args.guidance,
        'controlnet_scale': args.controlnet_scale,
        'depth_scale': args.depth_scale,
        'ip_adapter_scale': args.ip_scale,
        'strength': args.strength,
        'expand_arm_mask': args.expand_arm,
        'seed': args.seed,
        'device': args.device
    }
    

  
    run_ipadapter_controlnet(**kwargs)
