import cv2
import numpy as np
import json
import math
import os
from typing import List, Tuple, Optional


def load_keypoints(json_path: str) -> dict:
    """Loads keypoints from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def save_keypoints(keypoints: List[List[float]], filename: str):
    """Saves keypoints to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(keypoints, f, indent=4)
    print(f"정렬된 키포인트가 '{filename}'에 저장되었습니다.")


def denormalize_keypoints(keypoints: List[List[float]], img_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Denormalizes a list of [x, y] keypoint pairs to image coordinates."""
    height, width = img_shape[:2]
    result = []
    for point in keypoints:
        x = int(point[0] * width)
        y = int(point[1] * height)
        result.append((x, y))
    return result


def normalize_keypoints(keypoints: List[Tuple[int, int]], img_shape: Tuple[int, int]) -> List[List[float]]:
    """Normalizes pixel coordinates back to [0, 1] range."""
    height, width = img_shape[:2]
    result = []
    for point in keypoints:
        x = float(point[0]) / width
        y = float(point[1]) / height
        result.append([x, y])
    return result


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

    # Find the sketch points closest to the original elbow and wrist
    if not sketch_pixels:
        print("스케치 픽셀이 없습니다. 정렬을 건너뜁니다.")
        return original_keypoints

    closest_elbow_pixel = min(sketch_pixels, key=lambda p: distance(p, elbow))
    closest_wrist_pixel = min(sketch_pixels, key=lambda p: distance(p, wrist))

    # Calculate original arm segment lengths
    original_upper_arm_length = distance(shoulder, elbow)
    original_forearm_length = distance(elbow, wrist)

    # New elbow point is the original shoulder moved along the direction to the closest sketch point
    upper_arm_direction = (closest_elbow_pixel[0] - shoulder[0], closest_elbow_pixel[1] - shoulder[1])
    upper_arm_length = distance(shoulder, closest_elbow_pixel)

    if upper_arm_length > 0:
        unit_vector_upper = (upper_arm_direction[0] / upper_arm_length, upper_arm_direction[1] / upper_arm_length)
        new_elbow = (
            int(shoulder[0] + unit_vector_upper[0] * original_upper_arm_length),
            int(shoulder[1] + unit_vector_upper[1] * original_upper_arm_length)
        )
    else:
        new_elbow = elbow

    # New wrist point is the new elbow moved along the direction to the closest sketch point
    forearm_direction = (closest_wrist_pixel[0] - new_elbow[0], closest_wrist_pixel[1] - new_elbow[1])
    forearm_length = distance(new_elbow, closest_wrist_pixel)

    if forearm_length > 0:
        unit_vector_forearm = (forearm_direction[0] / forearm_length, forearm_direction[1] / forearm_length)
        new_wrist = (
            int(new_elbow[0] + unit_vector_forearm[0] * original_forearm_length),
            int(new_elbow[1] + unit_vector_forearm[1] * original_forearm_length)
        )
    else:
        new_wrist = wrist

    # Update new keypoints list
    new_keypoints[elbow_idx] = new_elbow
    new_keypoints[wrist_idx] = new_wrist

    return new_keypoints


def align(idx):
    """
    Main alignment function.
    Reads original image, sketch, and keypoints, then aligns the pose to the sketch.
    Saves the aligned keypoints as keypoints_aligned_{idx}.json
    """
    # Get base directory (flask_app/data or flask_app/static)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, '..', '..')
    flask_app_dir = os.path.join(model_dir, '..', 'flask_app')
    
    # File paths - using data folder structure (not static)
    # Try multiple extensions (.png, .jpeg, .jpg)
    upload_dir = os.path.join(flask_app_dir, 'data', 'uploads')
    original_image_path = None
    for ext in ['.png', '.jpeg', '.jpg', '.PNG', '.JPEG', '.JPG']:
        test_path = os.path.join(upload_dir, f'{idx}{ext}')
        if os.path.exists(test_path):
            original_image_path = test_path
            break
    
    if original_image_path is None:
        raise FileNotFoundError(f"원본 이미지를 찾을 수 없습니다: {upload_dir}/{idx}.*")
    
    sketch_image_path = os.path.join(flask_app_dir, 'data', 'sketch', f'sketch_{idx}.png')
    keypoints_input_path = os.path.join(model_dir, 'pose', 'pose_json', f'keypoints_{idx}.json')
    keypoints_output_path = os.path.join(model_dir, 'pose', 'pose_json', f'keypoints_aligned_{idx}.json')

    print(f"Loading files for idx={idx}...")
    print(f"  Image: {original_image_path}")
    print(f"  Sketch: {sketch_image_path}")
    print(f"  Keypoints: {keypoints_input_path}")

    # Load images
    original_img = cv2.imread(original_image_path)
    sketch_img = cv2.imread(sketch_image_path)

    if original_img is None:
        raise FileNotFoundError(f"원본 이미지를 읽을 수 없습니다: {original_image_path}")
    if sketch_img is None:
        raise FileNotFoundError(f"스케치 이미지를 찾을 수 없습니다: {sketch_image_path}")

    # Load keypoints (expecting pixel coordinates)
    try:
        keypoints_data = load_keypoints(keypoints_input_path)
        # Assuming the JSON structure has a list of keypoints directly
        if isinstance(keypoints_data, list):
            body_keypoints_raw = keypoints_data
        elif 'bodies' in keypoints_data:
            body_keypoints_raw = keypoints_data['bodies']
        else:
            body_keypoints_raw = keypoints_data
            
        # Convert to tuple format for processing (already in pixel coords)
        original_keypoints = [(int(kpt[0]), int(kpt[1])) for kpt in body_keypoints_raw]
        
        if len(original_keypoints) < 18:
            print(f"경고: 키포인트가 {len(original_keypoints)}개입니다. 18개의 OpenPose 키포인트를 예상했습니다.")
    except (FileNotFoundError, KeyError, IndexError) as e:
        raise Exception(f"키포인트 파일을 읽는 중 문제가 발생했습니다: {e}")

    # Extract sketch information
    sketch_pixels = get_sketch_red_pixels(sketch_img)
    if not sketch_pixels:
        print("스케치에서 빨간색 선을 찾지 못했습니다. 원본 키포인트를 그대로 사용합니다.")
        aligned_keypoints = original_keypoints
    else:
        # Find the starting point of the sketch line (closest to a shoulder)
        sketch_start_point = min(sketch_pixels, key=lambda p: math.sqrt(
            (p[0] - original_keypoints[2][0]) ** 2 + (p[1] - original_keypoints[2][1]) ** 2))

        # Determine which arm to align
        arm_to_align = find_closest_arm(sketch_start_point, original_keypoints)

        # Align the chosen arm to the sketch curve
        aligned_keypoints = align_arm_to_sketch(original_keypoints, sketch_pixels, arm_to_align)

    # Convert back to list format (keep as pixel coordinates)
    aligned_keypoints_list = [[float(kpt[0]), float(kpt[1])] for kpt in aligned_keypoints]

    # Save the aligned keypoints (in pixel coordinates)
    save_keypoints(aligned_keypoints_list, keypoints_output_path)
    print(f"Alignment completed for idx={idx}")
    print(f"Output saved to: {keypoints_output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Align pose keypoints to sketch')
    parser.add_argument('--idx', type=str, required=True, help='Image index (e.g., "1", "2")')
    args = parser.parse_args()
    
    try:
        align(args.idx)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
