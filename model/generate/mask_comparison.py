"""마스크 블러/페더링 설정 비교 테스트"""
import subprocess
import os
import shutil
from PIL import Image, ImageDraw, ImageFont
import numpy as np

mask_test_dir = "/workspace/AniSLE/flask_app/data/results/mask_test"
os.makedirs(mask_test_dir, exist_ok=True)

mask_configs = [
    (31, 40, 15, "blur31_f40_b15"),
    (41, 50, 21, "blur41_f50_b21_current"),
    (51, 60, 25, "blur51_f60_b25"),
    (61, 70, 31, "blur61_f70_b31"),
    (71, 80, 35, "blur71_f80_b35"),
    (81, 90, 41, "blur81_f90_b41"),
]

# Base command
base_cmd = [
    "python", "ipadapter_demo.py",
    "--idx", "28",
    "--controlnet-scale", "1.2",
    "--ip-scale", "0.4",
    "--guidance", "3.3",
    "--steps", "25",
    "--strength", "1.0",
    "--no-expand-arm"
]

# Temporarily backup original file
original_file = "/workspace/AniSLE/model/generate/ipadapter_demo.py"
backup_file = "/workspace/AniSLE/model/generate/ipadapter_demo.py.mask_backup"
shutil.copy2(original_file, backup_file)

output_images = []

print("Starting mask smoothing comparison tests...")
print("=" * 60)

try:
    for blur1, feather, blur2, label in mask_configs:
        print(f"\nTesting: {label}")
        print(f"  First blur: {blur1}x{blur1}, Feather: {feather}px, Final blur: {blur2}x{blur2}")
        
        # Read original file
        with open(backup_file, 'r') as f:
            content = f.read()
        
        # Replace mask processing parameters
        # Find and replace the three key lines
        old_blur1 = "mask_array = cv2.GaussianBlur(mask_array, (41, 41), 0)"
        new_blur1 = f"mask_array = cv2.GaussianBlur(mask_array, ({blur1}, {blur1}), 0)"
        
        old_feather = "feather_width = 50"
        new_feather = f"feather_width = {feather}"
        
        old_blur2 = "mask_normalized = cv2.GaussianBlur(mask_normalized, (21, 21), 0)"
        new_blur2 = f"mask_normalized = cv2.GaussianBlur(mask_normalized, ({blur2}, {blur2}), 0)"
        
        content = content.replace(old_blur1, new_blur1)
        content = content.replace(old_feather, new_feather)
        content = content.replace(old_blur2, new_blur2)
        
        # Write modified file
        with open(original_file, 'w') as f:
            f.write(content)
        
        # Run the command
        result = subprocess.run(base_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ {label} completed successfully")
            output_path = "/workspace/AniSLE/flask_app/data/results/28_ipadapter.png"
            if os.path.exists(output_path):
                # Save individual image to mask_test folder
                individual_save_path = os.path.join(mask_test_dir, f"{label}.png")
                shutil.copy2(output_path, individual_save_path)
                print(f"  Saved to: {individual_save_path}")
                
                img = Image.open(output_path)
                # Add text label
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
                except:
                    font = ImageFont.load_default()
                
                # Add label with background for visibility
                label_text = f"B1:{blur1} F:{feather} B2:{blur2}"
                bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                draw.rectangle([(5, 5), (15 + text_width, 15 + text_height)], fill="black")
                draw.text((10, 10), label_text, fill="yellow", font=font)
                
                output_images.append(img)
            else:
                print(f"✗ Output file not found for {label}")
        else:
            print(f"✗ Failed for {label}")
            print(result.stderr)
    
finally:
    # Restore original file
    print("\nRestoring original file...")
    shutil.copy2(backup_file, original_file)
    os.remove(backup_file)

print("\n" + "=" * 60)
print(f"Collected {len(output_images)} images")

if output_images:
    # Create comparison grid (2 rows x 3 cols)
    img_width, img_height = output_images[0].size
    grid_width = img_width * 3
    grid_height = img_height * 2
    
    comparison = Image.new('RGB', (grid_width, grid_height), 'white')
    
    for idx, img in enumerate(output_images):
        row = idx // 3
        col = idx % 3
        x = col * img_width
        y = row * img_height
        comparison.paste(img, (x, y))
    
    output_path = os.path.join(mask_test_dir, "mask_comparison.png")
    comparison.save(output_path)
    print(f"\n✓ Comparison image saved to: {output_path}")
    print(f"  Grid layout: 2 rows x 3 columns")
    print(f"  Image size: {grid_width}x{grid_height}")
    print(f"\nIndividual images saved in: {mask_test_dir}/")
else:
    print("\n✗ No images to compare")
