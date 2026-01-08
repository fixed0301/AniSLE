"""마스크 dilation 변화 테스트 (blur71 기반)"""
import subprocess
import os
import shutil
from PIL import Image, ImageDraw, ImageFont
import numpy as np

mask_test_dir = "/workspace/AniSLE/flask_app/data/results/mask_test"
os.makedirs(mask_test_dir, exist_ok=True)

mask_configs = [
    (71, 80, 35, 0, "blur71_dilate0_baseline"),
    (71, 80, 35, 3, "blur71_dilate3"),
    (71, 80, 35, 5, "blur71_dilate5"),
    (71, 80, 35, 7, "blur71_dilate7"),
    (71, 80, 35, 9, "blur71_dilate9"),
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

print("Starting mask dilate comparison tests (blur71 base)...")
print("=" * 60)

try:
    for blur1, feather, blur2, dilate_size, label in mask_configs:
        print(f"\nTesting: {label}")
        print(f"  Blur: {blur1}x{blur1}, Feather: {feather}px, Final blur: {blur2}x{blur2}, Dilate: {dilate_size}")
        
        # Read original file
        with open(backup_file, 'r') as f:
            content = f.read()
        
        # Replace mask processing parameters
        old_blur1 = "mask_array = cv2.GaussianBlur(mask_array, (41, 41), 0)"
        new_blur1 = f"mask_array = cv2.GaussianBlur(mask_array, ({blur1}, {blur1}), 0)"
        
        old_feather = "feather_width = 50"
        new_feather = f"feather_width = {feather}"
        
        old_blur2 = "mask_normalized = cv2.GaussianBlur(mask_normalized, (21, 21), 0)"
        new_blur2 = f"mask_normalized = cv2.GaussianBlur(mask_normalized, ({blur2}, {blur2}), 0)"
        
        content = content.replace(old_blur1, new_blur1)
        content = content.replace(old_feather, new_feather)
        content = content.replace(old_blur2, new_blur2)
        
        # Add dilate operation if dilate_size > 0
        if dilate_size > 0:
            # Find the location to insert dilate code (after first blur)
            insert_marker = f"mask_array = cv2.GaussianBlur(mask_array, ({blur1}, {blur1}), 0)\n    Image.fromarray(mask_array).save(os.path.join(debug_dir, 'debug_mask_blurred.png'))"
            dilate_code = f"mask_array = cv2.GaussianBlur(mask_array, ({blur1}, {blur1}), 0)\n    # Apply slight dilation\n    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ({dilate_size}, {dilate_size}))\n    mask_array = cv2.dilate(mask_array, kernel, iterations=1)\n    Image.fromarray(mask_array).save(os.path.join(debug_dir, 'debug_mask_blurred.png'))"
            content = content.replace(insert_marker, dilate_code)
        
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
                label_text = f"B:{blur1} F:{feather} D:{dilate_size}"
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
    # Create comparison grid (1 row x 5 cols or 2 rows depending on count)
    img_width, img_height = output_images[0].size
    num_cols = min(3, len(output_images))
    num_rows = (len(output_images) + num_cols - 1) // num_cols
    grid_width = img_width * num_cols
    grid_height = img_height * num_rows
    
    comparison = Image.new('RGB', (grid_width, grid_height), 'white')
    
    for idx, img in enumerate(output_images):
        row = idx // num_cols
        col = idx % num_cols
        x = col * img_width
        y = row * img_height
        comparison.paste(img, (x, y))
    
    output_path = os.path.join(mask_test_dir, "dilate_comparison.png")
    comparison.save(output_path)
    print(f"\n✓ Comparison image saved to: {output_path}")
    print(f"  Grid layout: {num_rows} rows x {num_cols} columns")
    print(f"  Image size: {grid_width}x{grid_height}")
    print(f"\nIndividual images saved in: {mask_test_dir}/")
else:
    print("\n✗ No images to compare")
