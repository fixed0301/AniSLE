from PIL import Image, ImageDraw
import numpy as np
from typing import List, Tuple, Optional
from align import load_keypoints

def draw_skeleton_on_image(img: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0),
                           thickness: int = 3, save = True):
    """Draws a skeleton on an image."""
    # Create a black background image if img is None or create from keypoints
    if img is None:
        img = Image.new("RGB", (256, 256), (0, 0, 0))
    else:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))

    keypoints = load_keypoints("../../keypoints_aligned.json")
    
    # Create a drawing context
    draw = ImageDraw.Draw(img)

    connections = [
        (2, 3), (3, 4),  # Right arm
        (5, 6), (6, 7),  # Left arm
        (1, 2), (1, 5),  # Neck to shoulders
        (1, 0),  # Neck to nose
        (1, 8), (8, 9), (9, 10),  # Torso and right hip/leg
        (1, 11), (11, 12), (12, 13),  # Torso and left hip/leg
        (0, 14), (0, 15),  # Nose to eyes
        (14, 16), (15, 17)  # Eyes to ears
    ]

    for start_idx, end_idx in connections:
        if len(keypoints) > max(start_idx, end_idx):
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]
            draw.line([start_point, end_point], fill=color, width=thickness)

    for i, point in enumerate(keypoints):
        # Draw circles as ellipses
        radius = 3
        draw.ellipse(
            [(point[0] - radius, point[1] - radius), 
             (point[0] + radius, point[1] + radius)], 
            fill=color
        )
    
    if save:
        img.save("skeleton_image.png")
    
    return img

draw_skeleton_on_image(None)

