#!/usr/bin/env python3
"""IP-Adapter 파라미터 스윕 배치 테스트"""

import os
import subprocess
from PIL import Image, ImageDraw, ImageFont
import numpy as np

idx = 28
controlnet_scale = 1.2
strength = 1.0
steps = 40

ip_scales = [0.3, 0.4, 0.5, 0.6]
guidances = [3.4, 3.6, 3.8, 4.0]

output_dir = "../../flask_app/data/results"
temp_dir = os.path.join(output_dir, "batch_test")
os.makedirs(temp_dir, exist_ok=True)

print(f"Starting batch test with {len(ip_scales)} x {len(guidances)} = {len(ip_scales) * len(guidances)} combinations")
print(f"ip-scale: {ip_scales}")
print(f"guidance: {guidances}")
print(f"Fixed params: controlnet-scale={controlnet_scale}, strength={strength}, steps={steps}")
print("="*80)

# Run tests
results = []
for i, ip_scale in enumerate(ip_scales):
    for j, guidance in enumerate(guidances):
        print(f"\n[{i*len(guidances)+j+1}/{len(ip_scales)*len(guidances)}] Testing ip-scale={ip_scale}, guidance={guidance}")
        
        # Run generation
        cmd = [
            "python", "ipadapter_demo.py",
            "--idx", str(idx),
            "--controlnet-scale", str(controlnet_scale),
            "--ip-scale", str(ip_scale),
            "--guidance", str(guidance),
            "--steps", str(steps),
            "--strength", str(strength),
            "--no-expand-arm"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Copy result to temp with labeled filename
            src_path = os.path.join(output_dir, f"{idx}_ipadapter.png")
            dst_filename = f"ip{ip_scale:.1f}_g{guidance:.1f}.png"
            dst_path = os.path.join(temp_dir, dst_filename)
            
            if os.path.exists(src_path):
                img = Image.open(src_path)
                img.save(dst_path)
                results.append({
                    'path': dst_path,
                    'ip_scale': ip_scale,
                    'guidance': guidance,
                    'filename': dst_filename
                })
                print(f"  ✓ Saved to {dst_filename}")
            else:
                print(f"  ✗ Output file not found: {src_path}")
        else:
            print(f"  ✗ Generation failed")
            print(result.stderr[-500:] if result.stderr else "")

print(f"\n{'='*80}")
print(f"Generated {len(results)} images successfully")
print("Creating comparison grid...")

# Create grid image
if results:
    # Load first image to get dimensions
    sample_img = Image.open(results[0]['path'])
    img_w, img_h = sample_img.size
    
    # Grid dimensions
    n_cols = len(guidances)
    n_rows = len(ip_scales)
    
    # Add space for labels
    label_height = 60
    label_width = 120
    
    # Create grid canvas
    grid_w = label_width + img_w * n_cols
    grid_h = label_height + img_h * n_rows
    grid_img = Image.new('RGB', (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid_img)
    
    # Try to load a font, fallback to default
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw column headers (guidance values)
    for j, guidance in enumerate(guidances):
        x = label_width + j * img_w + img_w // 2
        y = label_height // 2
        text = f"G={guidance:.1f}"
        bbox = draw.textbbox((0, 0), text, font=font_small)
        text_w = bbox[2] - bbox[0]
        draw.text((x - text_w//2, y - 10), text, fill=(0, 0, 0), font=font_small)
    
    # Draw row headers (ip_scale values)
    for i, ip_scale in enumerate(ip_scales):
        x = label_width // 2
        y = label_height + i * img_h + img_h // 2
        text = f"IP={ip_scale:.1f}"
        bbox = draw.textbbox((0, 0), text, font=font_small)
        text_w = bbox[2] - bbox[0]
        draw.text((x - text_w//2, y - 10), text, fill=(0, 0, 0), font=font_small)
    
    # Place images in grid
    for result in results:
        ip_scale = result['ip_scale']
        guidance = result['guidance']
        
        i = ip_scales.index(ip_scale)
        j = guidances.index(guidance)
        
        img = Image.open(result['path'])
        x = label_width + j * img_w
        y = label_height + i * img_h
        
        grid_img.paste(img, (x, y))
    
    # Add title
    title = f"Parameter Sweep: IP-Scale vs Guidance (CN={controlnet_scale}, Str={strength}, Steps={steps})"
    bbox = draw.textbbox((0, 0), title, font=font_large)
    title_w = bbox[2] - bbox[0]
    draw.text((grid_w//2 - title_w//2, 10), title, fill=(0, 0, 0), font=font_large)
    
    # Save grid
    grid_path = os.path.join(output_dir, f"comparison_grid_{idx}.png")
    grid_img.save(grid_path)
    print(f"\n✓ Comparison grid saved to: {grid_path}")
    print(f"  Grid size: {grid_w}x{grid_h} pixels")
    print(f"  Layout: {n_rows} rows x {n_cols} columns")
    
    # Also save a summary text file
    summary_path = os.path.join(output_dir, f"comparison_summary_{idx}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Parameter Sweep Results\n")
        f.write(f"="*80 + "\n\n")
        f.write(f"Fixed parameters:\n")
        f.write(f"  - idx: {idx}\n")
        f.write(f"  - controlnet_scale: {controlnet_scale}\n")
        f.write(f"  - strength: {strength}\n")
        f.write(f"  - steps: {steps}\n\n")
        f.write(f"Tested parameters:\n")
        f.write(f"  - ip_scale: {ip_scales}\n")
        f.write(f"  - guidance: {guidances}\n\n")
        f.write(f"Results ({len(results)} images):\n")
        for result in sorted(results, key=lambda x: (x['ip_scale'], x['guidance'])):
            f.write(f"  - ip_scale={result['ip_scale']:.1f}, guidance={result['guidance']:.1f} -> {result['filename']}\n")
    
    print(f"✓ Summary saved to: {summary_path}")
    print(f"\nDone! Check {grid_path} to compare results.")
else:
    print("No images generated successfully.")
