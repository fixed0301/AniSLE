"""마스크 생성 로직 테스트 및 검증"""
import os
import sys
import cv2
import numpy as np
from PIL import Image
import json

model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, model_dir)

idx = 33

source_path = f'/workspace/AniSLE/flask_app/data/uploads/{idx}.jpeg'
sketch_path = f'/workspace/AniSLE/flask_app/data/sketch/sketch_{idx}.png'
keypoints_path = f'{model_dir}/pose/pose_json/keypoints_{idx}.json'

print(f"Testing mask generation for idx={idx}")
print("=" * 60)

# 1. Load source image
source_img = Image.open(source_path).convert('RGB')
print(f"✓ Source image loaded: {source_img.size}")

# 2. Load keypoints
with open(keypoints_path, 'r') as f:
    keypoints_data = json.load(f)
    keypoints = keypoints_data['keypoints']
print(f"✓ Original keypoints loaded: {len(keypoints)} points")
print(f"  Sample keypoints: {keypoints[:3]}")

# 3. Load sketch
sketch_img = Image.open(sketch_path).convert('RGBA')
sketch_np = np.array(sketch_img)
print(f"✓ Sketch loaded: {sketch_img.size}")

# 4. Convert sketch to BGR for OpenCV
sketch_bgr = cv2.cvtColor(sketch_np, cv2.COLOR_RGBA2BGR)

# 5. Get sketch red pixels (assuming align module)
sys.path.insert(0, model_dir)
import align

sketch_pixels = align.get_sketch_red_pixels(sketch_bgr)
print(f"✓ Sketch pixels extracted: {len(sketch_pixels)} pixels")

if sketch_pixels:
    sketch_start = min(sketch_pixels, key=lambda p: np.sqrt((p[0] - keypoints[2][0])**2 + (p[1] - keypoints[2][1])**2))
    arm_to_align = align.find_closest_arm(sketch_start, keypoints)
    aligned_keypoints = align.align_arm_to_sketch(keypoints, sketch_pixels, arm_to_align)
    print(f"✓ Arm aligned: {arm_to_align}")
else:
    print("⚠ No sketch pixels found!")
    aligned_keypoints = keypoints
    arm_to_align = 'right'

# 6. Generate mask with IMPROVED logic
print("\n" + "=" * 60)
print("Generating mask with IMPROVED settings...")
print("=" * 60)

sketch_alpha = np.array(sketch_img)[:, :, 3]
sketch_mask = (sketch_alpha > 10).astype(np.uint8) * 255
print(f"Sketch mask: {(sketch_mask > 128).sum() / sketch_mask.size * 100:.1f}% coverage")

original_arm_mask = np.zeros((1024, 1024), dtype=np.uint8)

if arm_to_align == 'right':
    arm_keypoints = [keypoints[2], keypoints[3], keypoints[4]]
elif arm_to_align == 'left':
    arm_keypoints = [keypoints[5], keypoints[6], keypoints[7]]
else:
    arm_keypoints = []

for i in range(len(arm_keypoints) - 1):
    pt1 = tuple(map(int, arm_keypoints[i]))
    pt2 = tuple(map(int, arm_keypoints[i + 1]))
    cv2.line(original_arm_mask, pt1, pt2, 255, thickness=180)

for pt in arm_keypoints:
    cv2.circle(original_arm_mask, tuple(map(int, pt)), 100, 255, -1)

print(f"Original arm mask: {(original_arm_mask > 128).sum() / original_arm_mask.size * 100:.1f}% coverage")

aligned_arm_mask = np.zeros((1024, 1024), dtype=np.uint8)

if arm_to_align == 'right':
    aligned_arm_keypoints = [aligned_keypoints[2], aligned_keypoints[3], aligned_keypoints[4]]
elif arm_to_align == 'left':
    aligned_arm_keypoints = [aligned_keypoints[5], aligned_keypoints[6], aligned_keypoints[7]]
else:
    aligned_arm_keypoints = []

for i in range(len(aligned_arm_keypoints) - 1):
    pt1 = tuple(map(int, aligned_arm_keypoints[i]))
    pt2 = tuple(map(int, aligned_arm_keypoints[i + 1]))
    cv2.line(aligned_arm_mask, pt1, pt2, 255, thickness=180)

for pt in aligned_arm_keypoints:
    cv2.circle(aligned_arm_mask, tuple(map(int, pt)), 100, 255, -1)

print(f"Aligned arm mask: {(aligned_arm_mask > 128).sum() / aligned_arm_mask.size * 100:.1f}% coverage")

combined_mask = np.maximum(original_arm_mask, aligned_arm_mask)
print(f"Combined mask (original + aligned arm only): {(combined_mask > 128).sum() / combined_mask.size * 100:.1f}% coverage")

# Save intermediate for debug
debug_dir = '/workspace/AniSLE/flask_app/data/results/mask_debug'
os.makedirs(debug_dir, exist_ok=True)

Image.fromarray(sketch_mask).save(os.path.join(debug_dir, 'step1_sketch_mask.png'))
Image.fromarray(original_arm_mask).save(os.path.join(debug_dir, 'step2_original_arm_mask.png'))
Image.fromarray(aligned_arm_mask).save(os.path.join(debug_dir, 'step3_aligned_arm_mask.png'))
Image.fromarray(combined_mask).save(os.path.join(debug_dir, 'step4_combined_mask.png'))

print("\nApplying post-processing (NO feathering)...")

mask_blurred = cv2.GaussianBlur(combined_mask, (101, 101), 0)
Image.fromarray(mask_blurred).save(os.path.join(debug_dir, 'step5_blur101.png'))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
mask_dilated = cv2.dilate(mask_blurred, kernel, iterations=2)
print(f"After dilation: {(mask_dilated > 128).sum() / mask_dilated.size * 100:.1f}% coverage")
Image.fromarray(mask_dilated).save(os.path.join(debug_dir, 'step6_dilate25.png'))

mask_normalized = mask_dilated.astype(np.float32) / 255.0
mask_normalized = cv2.GaussianBlur(mask_normalized, (71, 71), 0)
mask_final = (mask_normalized * 255).astype(np.uint8)

print(f"Final mask: {(mask_final > 128).sum() / mask_final.size * 100:.1f}% coverage")
Image.fromarray(mask_final).save(os.path.join(debug_dir, 'step8_final_blur45.png'))

# Create visualization with all steps
print("\n" + "=" * 60)
print("Creating comparison visualization...")
print("=" * 60)

# Load all debug images
steps = [
    ('1. Sketch (Not Used)', Image.open(os.path.join(debug_dir, 'step1_sketch_mask.png'))),
    ('2. Original Arm', Image.open(os.path.join(debug_dir, 'step2_original_arm_mask.png'))),
    ('3. Aligned Arm', Image.open(os.path.join(debug_dir, 'step3_aligned_arm_mask.png'))),
    ('4. Combined (2+3)', Image.open(os.path.join(debug_dir, 'step4_combined_mask.png'))),
    ('5. Blur 101x101', Image.open(os.path.join(debug_dir, 'step5_blur101.png'))),
    ('6. Dilate 25x25', Image.open(os.path.join(debug_dir, 'step6_dilate25.png'))),
    ('7. Final Blur 71x71', Image.fromarray(mask_final)),
    ('8. Overlay on Source', None),  # Will create below
]

# Create overlay image for step 8
source_rgb = source_img.resize((1024, 1024))
overlay = source_rgb.copy()
overlay_np = np.array(overlay)
mask_colored = cv2.applyColorMap(mask_final, cv2.COLORMAP_JET)
mask_colored_rgb = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
overlay_np = cv2.addWeighted(overlay_np, 0.6, mask_colored_rgb, 0.4, 0)
steps[7] = ('8. Overlay on Source', Image.fromarray(overlay_np))

# Create grid (2 rows x 4 cols)
from PIL import ImageDraw, ImageFont
grid_img = Image.new('RGB', (1024*4, 1024*2), 'white')

for idx, (label, img) in enumerate(steps):
    row = idx // 4
    col = idx % 4
    x = col * 1024
    y = row * 1024
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Add label
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Background for text
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.rectangle([(10, 10), (20 + text_width, 20 + text_height)], fill='black')
    draw.text((15, 15), label, fill='yellow', font=font)
    
    grid_img.paste(img, (x, y))

grid_path = os.path.join(debug_dir, 'mask_generation_comparison.png')
grid_img.save(grid_path)

print(f"\n✅ All debug images saved to: {debug_dir}/")
print(f"✅ Comparison grid saved to: {grid_path}")
print(f"\n📊 Final mask coverage: {(mask_final > 128).sum() / mask_final.size * 100:.2f}%")
print(f"📊 Settings used:")
print(f"   - Line thickness: 180 (was 80)")
print(f"   - Joint radius: 100 (was 50)")
print(f"   - Initial blur: 101x101 (was 71x71)")
print(f"   - Dilation kernel: 40x40 (was 15x15)")
print(f"   - Dilation iterations: 2 (was 1)")
print(f"   - Feathering: REMOVED (was 80px)")
print(f"   - Final blur: 71x71 (was 35x35)")
print(f"   - Mask source: Original + Aligned arm ONLY (sketch excluded)")
