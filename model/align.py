import cv2
import numpy as np
import json
import math
from typing import List, Tuple, Optional


def load_keypoints(json_path: str) -> dict:
    """Loads keypoints from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def denormalize_keypoints(keypoints: List[List[float]], img_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Denormalizes a list of [x, y] keypoint pairs to image coordinates."""
    height, width = img_shape[:2]
    result = []
    for point in keypoints:
        x = int(point[0] * width)
        y = int(point[1] * height)
        result.append((x, y))
    return result

def save_keypoints(keypoints: List[Tuple[int, int]], filename: str):
    """Saves keypoints to a JSON file."""
    # Keypoints are in pixel coordinates (integers).
    with open(filename, 'w') as f:
        json.dump(keypoints, f, indent=4)
    print(f"정렬된 키포인트가 '{filename}'에 저장되었습니다.")

def get_sketch_red_pixels(sketch_img: np.ndarray) -> List[Tuple[int, int]]:
    """Extracts all red pixels from the sketch image."""
    if len(sketch_img.shape) == 2:
        sketch_img = cv2.cvtColor(sketch_img, cv2.COLOR_GRAY2BGR)

    # Define a range for the red color
    lower_red = np.array([0, 0, 100])
    upper_red = np.array([50, 50, 255])

    # Create a mask for red pixels
    mask = cv2.inRange(sketch_img, lower_red, upper_red)
    red_pixels = cv2.findNonZero(mask)

    if red_pixels is None or len(red_pixels) < 2:
        return []

    # Reshape to a list of tuples
    return [tuple(p[0]) for p in red_pixels]


def find_closest_arm(sketch_start: Tuple[int, int], original_keypoints: List[Tuple[int, int]]) -> Optional[str]:
    """Determines if the sketch is for the left or right arm based on the closest shoulder."""
    if len(original_keypoints) < 8:
        print("키포인트가 부족하여 팔을 선택할 수 없습니다.")
        return None

    right_shoulder = original_keypoints[2]
    left_shoulder = original_keypoints[5]

    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    dist_right = distance(sketch_start, right_shoulder)
    dist_left = distance(sketch_start, left_shoulder)

    if dist_right < dist_left:
        print(f"오른쪽 팔 선택 (거리: {dist_right:.1f} vs {dist_left:.1f})")
        return "right"
    else:
        print(f"왼쪽 팔 선택 (거리: {dist_right:.1f} vs {dist_left:.1f})")
        return "left"


def align_arm_to_sketch(original_keypoints: List[Tuple[int, int]], sketch_pixels: List[Tuple[int, int]],
                        arm_to_align: str) -> List[Tuple[int, int]]:
    """Aligns the specified arm to the sketch curve, preserving original segment lengths."""

    new_keypoints = original_keypoints.copy()

    if arm_to_align == "right":
        shoulder_idx, elbow_idx, wrist_idx = 2, 3, 4
    elif arm_to_align == "left":
        shoulder_idx, elbow_idx, wrist_idx = 5, 6, 7
    else:
        return original_keypoints

    try:
        shoulder = original_keypoints[shoulder_idx]
        elbow = original_keypoints[elbow_idx]
        wrist = original_keypoints[wrist_idx]
    except IndexError:
        print("정렬할 팔의 키포인트가 부족합니다. 정렬을 건너뜁니다.")
        return original_keypoints

    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    if not sketch_pixels:
        print("스케치 픽셀이 없습니다. 정렬을 건너뜁니다.")
        return original_keypoints

    # Calculate original arm segment lengths
    original_upper_arm_length = distance(shoulder, elbow)
    original_forearm_length = distance(elbow, wrist)

    # Find sketch start and end points
    # Start: closest to shoulder
    # End: farthest from shoulder
    start_pixel = min(sketch_pixels, key=lambda p: distance(p, shoulder))
    end_pixel = max(sketch_pixels, key=lambda p: distance(p, shoulder))
    
    print(f"  Sketch: start={start_pixel}, end={end_pixel}, {len(sketch_pixels)} pixels")
    
    # Direction from shoulder toward the end of the sketch
    sketch_direction = (end_pixel[0] - start_pixel[0], end_pixel[1] - start_pixel[1])
    sketch_length = distance(start_pixel, end_pixel)
    
    if sketch_length > 0:
        # Unit vector along the sketch direction
        unit_vector = (sketch_direction[0] / sketch_length, sketch_direction[1] / sketch_length)
        
        # New elbow: move from shoulder along sketch direction by upper arm length
        new_elbow = (
            int(shoulder[0] + unit_vector[0] * original_upper_arm_length),
            int(shoulder[1] + unit_vector[1] * original_upper_arm_length)
        )
        
        # New wrist: continue from elbow along sketch direction by forearm length
        new_wrist = (
            int(new_elbow[0] + unit_vector[0] * original_forearm_length),
            int(new_elbow[1] + unit_vector[1] * original_forearm_length)
        )
    else:
        # Sketch is just a point
        new_elbow = start_pixel
        new_wrist = start_pixel

    # Update new keypoints list
    new_keypoints[elbow_idx] = new_elbow
    new_keypoints[wrist_idx] = new_wrist
    
    print(f"  Aligned arm: shoulder={shoulder}, elbow={new_elbow}, wrist={new_wrist}")

    return new_keypoints


def draw_skeleton_on_image(img: np.ndarray, keypoints: List[Tuple[int, int]], color: Tuple[int, int, int] = (0, 255, 0),
                           thickness: int = 3, use_openpose_colors: bool = False):
    """
    Draws a skeleton on an image.
    
    Args:
        img: Image to draw on
        keypoints: List of (x, y) keypoint coordinates
        color: Single color for all limbs (if use_openpose_colors=False)
        thickness: Line thickness
        use_openpose_colors: Use OpenPose colorful style
    """

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
    
    # OpenPose standard colors
    openpose_colors = [
        (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
        (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
        (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
        (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
        (255, 0, 170), (255, 0, 85)
    ]

    if use_openpose_colors:
        # Draw limbs with gradient/ellipse effect like OpenPose
        for i, (start_idx, end_idx) in enumerate(connections):
            if len(keypoints) > max(start_idx, end_idx):
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                
                limb_color = openpose_colors[i % len(openpose_colors)]
                
                # keypoints are (x, y) tuples
                # Calculate angle and length
                x1, y1 = start_point
                x2, y2 = end_point
                mX = (x1 + x2) / 2
                mY = (y1 + y2) / 2
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                angle = np.degrees(np.arctan2(y1 - y2, x1 - x2))
                
                if length > 0:
                    # cv2.ellipse2Poly expects (x, y) center
                    polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), thickness), int(angle), 0, 360, 1)
                    cv2.fillConvexPoly(img, polygon, limb_color)
        
        # Draw keypoints as white circles
        for i, point in enumerate(keypoints):
            if point[0] > 0 and point[1] > 0:
                cv2.circle(img, point, 4, (255, 255, 255), -1)
    else:
        # Original simple drawing
        for start_idx, end_idx in connections:
            if len(keypoints) > max(start_idx, end_idx):
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                cv2.line(img, start_point, end_point, color, thickness)

        for i, point in enumerate(keypoints):
            cv2.circle(img, point, 3, color, -1)


def main():
    # File paths
    original_image_path = "../flask_app/static/uploads/groundimg.jpeg"
    sketch_image_path = "../flask_app/static/sketch/sketch_groundimg.png"
    keypoints_path = "keypoints.json"

    # Load images and data
    original_img = cv2.imread(original_image_path)
    sketch_img = cv2.imread(sketch_image_path)

    if original_img is None or sketch_img is None:
        print("Error: Could not load images.")
        return

    # Load and normalize keypoints
    try:
        keypoints_data = load_keypoints(keypoints_path)
        body_keypoints_raw = keypoints_data['bodies']
        original_keypoints = denormalize_keypoints(body_keypoints_raw, original_img.shape)
        if len(original_keypoints) < 18:
            print(f"경고: 키포인트가 {len(original_keypoints)}개 밖에 없습니다. 18개의 OpenPose 키포인트를 예상했습니다.")
            return
    except (FileNotFoundError, KeyError, IndexError) as e:
        print(f"오류: 키포인트 파일을 읽는 중 문제가 발생했습니다 - {e}")
        return

    # Extract sketch information
    sketch_pixels = get_sketch_red_pixels(sketch_img)
    if not sketch_pixels:
        print("스케치에서 빨간색 선을 찾지 못했습니다.")
        return

    # Find the starting point of the sketch line (closest to a shoulder)
    sketch_start_point = min(sketch_pixels, key=lambda p: math.sqrt(
        (p[0] - original_keypoints[2][0]) ** 2 + (p[1] - original_keypoints[2][1]) ** 2))

    # Determine which arm to align
    arm_to_align = find_closest_arm(sketch_start_point, original_keypoints)

    # Align the chosen arm to the sketch curve
    aligned_keypoints = align_arm_to_sketch(original_keypoints, sketch_pixels, arm_to_align)

    # Save the aligned keypoints to a new JSON file
    save_keypoints(aligned_keypoints, "keypoints_aligned.json")

    # Create the result image
    result_img = original_img.copy()

    # Draw original skeleton (Red)
    draw_skeleton_on_image(result_img, original_keypoints, (0, 0, 255), 2)

    # Draw aligned skeleton (OpenPose color style)
    draw_skeleton_on_image(result_img, aligned_keypoints, (0, 255, 0), 3, use_openpose_colors=True)
    
    # Also save aligned skeleton on black background
    black_bg = np.zeros_like(original_img)
    draw_skeleton_on_image(black_bg, aligned_keypoints, (0, 255, 0), 4, use_openpose_colors=True)
    cv2.imwrite("aligned_pose_only.png", black_bg)
    print("OpenPose 스타일 aligned 포즈가 'aligned_pose_only.png'로 저장되었습니다.")

    # Draw the sketch line for reference (Blue)
    # for pixel in sketch_pixels:
    #     cv2.circle(result_img, pixel, 1, (255, 0, 0), -1)

    # Add legend
    cv2.putText(result_img, "Red: Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(result_img, "OpenPose: Aligned", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    #cv2.putText(result_img, "Blue: Sketch", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Save and display
    cv2.imwrite("aligned_result.jpg", result_img)
    print("결과가 'aligned_result.jpg'로 저장되었습니다.")

    cv2.imshow("Aligned Skeleton", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# hand keypoint도 반영하도록

if __name__ == "__main__":
    main()