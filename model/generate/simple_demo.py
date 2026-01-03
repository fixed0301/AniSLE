"""
Simple PCDMs Demo for zeroshot folder
Based on pcdms_demo.ipynb - simplified for easy use
"""

import os
import sys
import torch
from PIL import Image, ImageDraw
import numpy as np

# Add current directory to path for local imports (AniSLE structure)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Go up to model/ directory
sys.path.insert(0, current_dir)

from src.models.stage2_inpaint_unet_2d_condition import Stage2_InapintUNet2DConditionModel
from src.pipelines.PCDMs_pipeline import PCDMsPipeline
from single_extract_pose import inference_pose
from src.controlnet_aux.dwpose import draw_pose

import json

import torch.nn as nn
from diffusers.models.controlnet import ControlNetConditioningEmbedding
from transformers import Dinov2Model
from diffusers import AutoencoderKL, DDIMScheduler
from torchvision import transforms


class ImageProjModel(torch.nn.Module):
    """Image projection model"""
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def load_models(device="cuda", dtype=torch.float16):
    """Load PCDMs models"""
    print("Loading PCDMs models...")
    
    pretrained_model = "sd2-community/stable-diffusion-2-1"
    image_encoder_path = "facebook/dinov2-giant"
    model_ckpt_path = os.path.join(current_dir, "pcdms_ckpt.pt")
    
    # Load UNet
    unet = Stage2_InapintUNet2DConditionModel.from_pretrained(
        pretrained_model,
        subfolder="unet",
        torch_dtype=dtype,
        in_channels=9,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True
    ).to(device)
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        pretrained_model,
        subfolder="vae"
    ).to(device, dtype=dtype)
    
    # Load image encoder
    image_encoder = Dinov2Model.from_pretrained(image_encoder_path).to(device, dtype=dtype)
    
    # Create projection models
    image_proj_model = ImageProjModel(
        in_dim=1536,
        hidden_dim=768,
        out_dim=1024  # PCDMs uses 1024
    ).to(device, dtype=dtype)
    
    pose_proj_model = ControlNetConditioningEmbedding(
        conditioning_embedding_channels=320,
        block_out_channels=(16, 32, 96, 256),
        conditioning_channels=3
    ).to(device, dtype=dtype)
    
    # Load trained weights from checkpoint
    if os.path.exists(model_ckpt_path):
        print(f"Loading checkpoint from {model_ckpt_path}...")
        model_sd = torch.load(model_ckpt_path, map_location="cpu")["module"]
        
        image_proj_model_dict = {}
        pose_proj_dict = {}
        unet_dict = {}
        
        for k in model_sd.keys():
            if k.startswith("pose_proj"):
                pose_proj_dict[k.replace("pose_proj.", "")] = model_sd[k]
            elif k.startswith("image_proj_model"):
                image_proj_model_dict[k.replace("image_proj_model.", "")] = model_sd[k]
            elif k.startswith("unet"):
                unet_dict[k.replace("unet.", "")] = model_sd[k]
        
        image_proj_model.load_state_dict(image_proj_model_dict)
        pose_proj_model.load_state_dict(pose_proj_dict)
        unet.load_state_dict(unet_dict)
        print("✓ Checkpoint loaded successfully!")
    else:
        print(f"Warning: Checkpoint not found at {model_ckpt_path}")
    
    # Scheduler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    
    # Create pipeline
    pipe = PCDMsPipeline.from_pretrained(
        pretrained_model,
        unet=unet,
        torch_dtype=dtype,
        scheduler=noise_scheduler,
        feature_extractor=None,
        safety_checker=None
    ).to(device)
    
    print("Models loaded successfully!")
    return pipe, vae, image_encoder, image_proj_model, pose_proj_model


def run_inference(
    source_image_path,
    target_pose_path=None,
    target_pose_json=None,  # NEW: JSON keypoints path
    mask_path=None,
    output_path="data/output.png",
    device="cuda",
    num_steps=50,
    guidance_scale=2.0,
    pose_conditioning_scale=1.0,  # NEW: Control pose influence (higher = stronger pose)
    seed=42  # Random seed
):
    """
    Run PCDMs inference with optional mask
    
    Args:
        source_image_path: Source image path
        target_pose_path: Target image to extract pose from (or None)
        target_pose_json: JSON file with keypoints (overrides target_pose_path)
        mask_path: Optional mask (white=generate, black=keep source)
        output_path: Output save path
    """
    
    dtype = torch.float16 if device == "cuda" else torch.float32
    image_size = (512, 512)
    
    # Load models
    pipe, vae, image_encoder, image_proj_model, pose_proj_model = load_models(device, dtype)
    
    # Transforms
    from transformers import CLIPImageProcessor
    clip_image_processor = CLIPImageProcessor()
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print(f"\n{'='*50}")
    print("Running PCDMs Inference")
    print(f"{'='*50}")
    print(f"Source: {source_image_path}")
    if target_pose_json:
        print(f"Target pose: {target_pose_json} (JSON keypoints)")
    else:
        print(f"Target pose: {target_pose_path or 'Using source pose'}")
    print(f"Mask: {mask_path or 'None (full generation)'}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Pose conditioning scale: {pose_conditioning_scale}")
    print(f"Seed: {seed}")
    
    # Load source image
    s_img = Image.open(source_image_path).convert("RGB").resize(image_size, Image.BICUBIC)
    
    # Load user mask if provided
    user_mask = None
    if mask_path and os.path.exists(mask_path):
        print(f"Loading user mask from: {mask_path}")
        mask_img_full = Image.open(mask_path).convert('L').resize(image_size, Image.BICUBIC)
        user_mask = np.array(mask_img_full) / 255.0  # 1=use generated, 0=keep source
        print(f"  Mask region: {user_mask.mean()*100:.1f}% will be replaced with generated result")
    
    # Create input for PCDMs: source | black (full generation)
    # PCDMs generates a complete new image with target pose on the right
    black_image = Image.new("RGB", s_img.size, (0, 0, 0)).resize(image_size, Image.BICUBIC)
    
    s_img_t_mask = Image.new("RGB", (s_img.width * 2, s_img.height))
    s_img_t_mask.paste(s_img, (0, 0))
    s_img_t_mask.paste(black_image, (s_img.width, 0))
    print(f"  Input: [source image | black] for full generation with target pose")
    
    # Extract poses
    print("\nExtracting poses...")
    # Change to parent directory for pose extraction
    original_dir = os.getcwd()
    os.chdir(parent_dir)
    try:
        s_pose = inference_pose(
            os.path.join(original_dir, source_image_path) if not os.path.isabs(source_image_path) else source_image_path,
            image_size=(image_size[1], image_size[0])
        ).resize(image_size, Image.BICUBIC)
        
        if target_pose_json:
            # Load JSON keypoints and draw pose
            print(f"  Loading JSON keypoints from: {target_pose_json}")
            json_path = os.path.join(original_dir, target_pose_json) if not os.path.isabs(target_pose_json) else target_pose_json
            print(f"  Full JSON path: {json_path}")
            with open(json_path, 'r') as f:
                keypoints = json.load(f)
            
            print(f"  Loaded {len(keypoints)} keypoints")
            
            # Convert keypoints to pose format expected by draw_pose
            # Draw skeleton directly using PIL instead of DWpose format
            if isinstance(keypoints, list):
                print(f"  Drawing skeleton directly from {len(keypoints)} keypoints")
                
                # Use keypoints as-is (already at correct scale)
                scaled_keypoints = [[int(kp[0]), int(kp[1])] for kp in keypoints]
                print(f"  Using keypoints at target size: {image_size[0]}x{image_size[1]}")
                
                # Create black canvas
                pose_canvas = Image.new("RGB", image_size, (0, 0, 0))
                draw = ImageDraw.Draw(pose_canvas)
                
                # OpenPose body connections (18 keypoints)
                connections = [
                    (1, 2), (1, 5),  # Neck to shoulders
                    (2, 3), (3, 4),  # Right arm
                    (5, 6), (6, 7),  # Left arm
                    (1, 8),  # Neck to mid hip
                    (8, 9), (9, 10),  # Right hip and leg
                    (8, 12), (12, 13), (13, 14),  # Left hip and leg (corrected)
                    (1, 0),  # Neck to nose
                    (0, 14), (0, 15),  # Nose to eyes
                    (14, 16), (15, 17),  # Eyes to ears
                    (11, 12), # Left hip connection
                    (8, 11) # Mid hip to left hip
                ]
                
                # Draw connections
                color = (0, 255, 0)  # Green
                thickness = 3
                
                for start_idx, end_idx in connections:
                    if start_idx < len(scaled_keypoints) and end_idx < len(scaled_keypoints):
                        start_point = tuple(scaled_keypoints[start_idx])
                        end_point = tuple(scaled_keypoints[end_idx])
                        draw.line([start_point, end_point], fill=color, width=thickness)
                
                # Draw keypoints as circles
                radius = 5
                for point in scaled_keypoints[:18]:  # Only body keypoints
                    x, y = point[0], point[1]
                    draw.ellipse(
                        [(x - radius, y - radius), (x + radius, y + radius)],
                        fill=(255, 0, 0)  # Red
                    )
                
                t_pose = pose_canvas
                print(f"  Skeleton drawn successfully")
                
                # Save debug pose image (use original_dir for relative paths)
                debug_output_path = os.path.join(original_dir, output_path) if not os.path.isabs(output_path) else output_path
                debug_dir = os.path.dirname(debug_output_path) or original_dir
                os.makedirs(debug_dir, exist_ok=True)
                debug_pose_path = os.path.join(debug_dir, 'debug_target_pose.png')
                t_pose.save(debug_pose_path)
                print(f"  DEBUG: Target pose saved to {debug_pose_path}")
            else:
                print("Warning: Unsupported JSON format, using source pose")
                t_pose = s_pose
                
        elif target_pose_path:
            # Convert to absolute path if needed
            target_pose_abs = os.path.join(original_dir, target_pose_path) if not os.path.isabs(target_pose_path) else target_pose_path
            # Extract pose from target image
            t_pose = inference_pose(target_pose_abs, image_size=(image_size[1], image_size[0])).resize(image_size, Image.BICUBIC)
        else:
            # Use source pose as target
            t_pose = s_pose
    finally:
        os.chdir(original_dir)
    
    # Concatenate poses (source | target)
    st_pose = Image.new("RGB", (s_pose.width * 2, s_pose.height))
    st_pose.paste(s_pose, (0, 0))
    st_pose.paste(t_pose, (s_pose.width, 0))
    
    # Save debug concatenated pose (use absolute path)
    debug_output_path = os.path.join(original_dir, output_path) if not os.path.isabs(output_path) else output_path
    debug_dir = os.path.dirname(debug_output_path) or original_dir
    debug_st_pose_path = os.path.join(debug_dir, 'debug_st_pose.png')
    st_pose.save(debug_st_pose_path)
    print(f"DEBUG: Concatenated pose (source|target) saved to {debug_st_pose_path}")
    
    # Prepare inputs
    clip_s_img = clip_image_processor(images=s_img, return_tensors="pt").pixel_values
    vae_image = torch.unsqueeze(img_transform(s_img_t_mask), 0)
    cond_st_pose = torch.unsqueeze(img_transform(st_pose), 0)
    
    # Pipeline mask: always full generation (left=keep, right=generate all)
    # User mask will be used later for post-processing blend
    mask1 = torch.ones((1, 1, int(image_size[0] / 8), int(image_size[1] / 8))).to(device, dtype=dtype)
    mask0 = torch.zeros((1, 1, int(image_size[0] / 8), int(image_size[1] / 8))).to(device, dtype=dtype)
    mask = torch.cat([mask1, mask0], dim=3)  # Left=keep source, Right=generate all
    
    print("\nGenerating image...")
    with torch.inference_mode():
        # Encode pose
        cond_pose = pose_proj_model(cond_st_pose.to(dtype=dtype, device=device))
        
        # Encode source image
        simg_mask_latents = vae.encode(vae_image.to(device, dtype=dtype)).latent_dist.sample()
        simg_mask_latents = simg_mask_latents * 0.18215
        
        # Get image embeddings
        images_embeds = image_encoder(clip_s_img.to(device, dtype=dtype)).last_hidden_state
        image_prompt_embeds = image_proj_model(images_embeds)
        uncond_image_prompt_embeds = image_proj_model(torch.zeros_like(images_embeds))
        
        # Run pipeline
        output = pipe(
            simg_mask_latents=simg_mask_latents,
            mask=mask,
            cond_pose=cond_pose,
            prompt_embeds=image_prompt_embeds,
            negative_prompt_embeds=uncond_image_prompt_embeds,
            height=image_size[1],
            width=image_size[0] * 2,
            num_images_per_prompt=1,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_steps,
        ).images[-1]
    
    # Save full output for debugging (should be 1024x512)
    output.save(output_path.replace('.png', '_full.png'))
    print(f"DEBUG: Full output saved to {output_path.replace('.png', '_full.png')} (size: {output.size})")
    
    # Extract right half (full generated image with target pose)
    result = output.crop((image_size[0], 0, image_size[0] * 2, image_size[1]))
    
    # Save raw generated result for debugging
    result.save(output_path.replace('.png', '_raw.png'))
    print(f"DEBUG: Full generated result saved to {output_path.replace('.png', '_raw.png')}")
    
    # Apply mask blending if mask provided
    if user_mask is not None:
        from scipy.ndimage import gaussian_filter
        
        print("\nApplying mask blending with Gaussian blur...")
        
        # Apply Gaussian blur to mask for smooth transition
        blur_radius = 5  # pixels
        mask_blurred = gaussian_filter(user_mask, sigma=blur_radius)
        mask_blurred = np.clip(mask_blurred, 0, 1)
        
        # Expand to 3 channels
        mask_3ch = np.stack([mask_blurred] * 3, axis=2)
        
        source_array = np.array(s_img).astype(float)
        result_array = np.array(result).astype(float)
        
        # Blend: mask=1 (white) uses generated, mask=0 (black) keeps source
        blended = result_array * mask_3ch + source_array * (1 - mask_3ch)
        result = Image.fromarray(blended.astype(np.uint8))
        
        print(f"  Blended {user_mask.mean()*100:.1f}% from generated pose, {(1-user_mask.mean())*100:.1f}% from source")
        print(f"  Gaussian blur radius: {blur_radius} pixels for smooth edges")
    
    # Save result
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    result.save(output_path)
    print(f"\n✓ Saved to: {output_path}")
    
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PCDMs Simple Demo")
    parser.add_argument('--idx', required=True, help='Image index')
    parser.add_argument('--steps', type=int, default=75, help='Number of inference steps')
    parser.add_argument('--guidance', type=float, default=7.0, help='Guidance scale')
    parser.add_argument('--pose-scale', type=float, default=1.5, help='Pose conditioning scale')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    
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
    mask_path = f'../mask/mask_img/mask_{idx}.png' if os.path.exists(f'../mask/mask_img/mask_{idx}.png') else None
    output_path = f'../../flask_app/data/results/{idx}.png'
    
    run_inference(
        source_image_path=source_path,
        target_pose_json=pose_json_path,
        mask_path=mask_path,
        output_path=output_path,
        device=args.device,
        num_steps=args.steps,
        guidance_scale=args.guidance,
        pose_conditioning_scale=args.pose_scale,
        seed=args.seed
    )
