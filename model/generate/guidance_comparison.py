"""여러 guidance 값을 테스트하고 비교 이미지 생성"""
import subprocess
import os
import shutil
from PIL import Image
import numpy as np

guidance_test_dir = "/workspace/AniSLE/flask_app/data/results/guidance_test"
os.makedirs(guidance_test_dir, exist_ok=True)

# Test parameters
base_cmd = [
    "python", "ipadapter_demo.py",
    "--idx", "28",
    "--controlnet-scale", "1.2",
    "--ip-scale", "0.4",
    "--steps", "25",
    "--strength", "1.0",
    "--no-expand-arm"
]

guidance_values = [3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]
output_images = []

print("Starting guidance comparison tests...")
print("=" * 60)

for guidance in guidance_values:
    print(f"\nTesting guidance={guidance}")
    cmd = base_cmd + ["--guidance", str(guidance)]
    
    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ guidance={guidance} completed successfully")
        # The output should be in flask_app/data/results/
        output_path = f"/workspace/AniSLE/flask_app/data/results/28_ipadapter.png"
        if os.path.exists(output_path):
            # Save individual image to guidance_test folder
            individual_save_path = os.path.join(guidance_test_dir, f"guidance_{guidance}.png")
            shutil.copy2(output_path, individual_save_path)
            print(f"  Saved to: {individual_save_path}")
            
            img = Image.open(output_path)
            # Add text label
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
            except:
                font = ImageFont.load_default()
            draw.text((10, 10), f"guidance={guidance}", fill="red", font=font)
            output_images.append(img)
        else:
            print(f"✗ Output file not found for guidance={guidance}")
    else:
        print(f"✗ Failed for guidance={guidance}")
        print(result.stderr)

print("\n" + "=" * 60)
print(f"Collected {len(output_images)} images")

if output_images:
    # Create comparison grid (2 rows x 4 cols)
    img_width, img_height = output_images[0].size
    grid_width = img_width * 4
    grid_height = img_height * 2
    
    comparison = Image.new('RGB', (grid_width, grid_height), 'white')
    
    for idx, img in enumerate(output_images):
        row = idx // 4
        col = idx % 4
        x = col * img_width
        y = row * img_height
        comparison.paste(img, (x, y))
    
    output_path = os.path.join(guidance_test_dir, "guidance_comparison.png")
    comparison.save(output_path)
    print(f"\n✓ Comparison image saved to: {output_path}")
    print(f"  Grid layout: 2 rows x 4 columns")
    print(f"  Image size: {grid_width}x{grid_height}")
else:
    print("\n✗ No images to compare")
