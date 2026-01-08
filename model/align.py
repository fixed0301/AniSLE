import cv2
import numpy as np
import json
import math
from typing import List, Tuple, Optional


def load_keypoints(json_path: str) -> dict:
    with open(json_path, 'r') as f:
        return json.load(f)


def denormalize_keypoints(keypoints: List[List[float]], img_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """normalized keypoints를 이미지 픽셀 좌표로 변환"""
    height, width = img_shape[:2]
    result = []
    for point in keypoints:
        x = int(point[0] * width)
        y = int(point[1] * height)
        result.append((x, y))
    return result

def save_keypoints(keypoints: List[Tuple[int, int]], filename: str):
    with open(filename, 'w') as f:
        json.dump(keypoints, f, indent=4)
    print(f"정렬된 키포인트가 '{filename}'에 저장되었습니다.")

def get_sketch_red_pixels(sketch_img: np.ndarray) -> List[Tuple[int, int]]:
    """스케치 이미지에서 빨간색 픽셀 좌표 추출"""
    if len(sketch_img.shape) == 2:
        sketch_img = cv2.cvtColor(sketch_img, cv2.COLOR_GRAY2BGR)

    lower_red = np.array([0, 0, 100])
    upper_red = np.array([50, 50, 255])
    mask = cv2.inRange(sketch_img, lower_red, upper_red)
    red_pixels = cv2.findNonZero(mask)

    if red_pixels is None or len(red_pixels) < 2:
        return []
    return [tuple(p[0]) for p in red_pixels]


def find_closest_arm(sketch_start: Tuple[int, int], original_keypoints: List[Tuple[int, int]]) -> Optional[str]:
    """스케치 시작점과 가장 가까운 어깨를 기준으로 좌우 팔 결정"""
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
    """스케치 곡선에 맞춰 팔 정렬 (원본 팔 길이 유지)"""
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

    # 원본 팔 길이 계산
    original_upper_arm_length = distance(shoulder, elbow)
    original_forearm_length = distance(elbow, wrist)

    # 스케치 시작/끝점 찾기 (어깨 기준 가장 가깝고/먼 점)
    start_pixel = min(sketch_pixels, key=lambda p: distance(p, shoulder))
    end_pixel = max(sketch_pixels, key=lambda p: distance(p, shoulder))
    print(f"  Sketch: start={start_pixel}, end={end_pixel}, {len(sketch_pixels)} pixels")
    
    # 스케치 방향으로 팔 재배치
    sketch_direction = (end_pixel[0] - start_pixel[0], end_pixel[1] - start_pixel[1])
    sketch_length = distance(start_pixel, end_pixel)
    
    if sketch_length > 0:
        unit_vector = (sketch_direction[0] / sketch_length, sketch_direction[1] / sketch_length)
        new_elbow = (
            int(shoulder[0] + unit_vector[0] * original_upper_arm_length),
            int(shoulder[1] + unit_vector[1] * original_upper_arm_length)
        )
        new_wrist = (
            int(new_elbow[0] + unit_vector[0] * original_forearm_length),
            int(new_elbow[1] + unit_vector[1] * original_forearm_length)
        )
    else:
        new_elbow = start_pixel
        new_wrist = start_pixel

    new_keypoints[elbow_idx] = new_elbow
    new_keypoints[wrist_idx] = new_wrist
    print(f"  Aligned arm: shoulder={shoulder}, elbow={new_elbow}, wrist={new_wrist}")

    return new_keypoints


def draw_skeleton_on_image(img: np.ndarray, keypoints: List[Tuple[int, int]], color: Tuple[int, int, int] = (0, 255, 0),
                           thickness: int = 3, use_openpose_colors: bool = False):
    """이미지에 스켈레톤 그리기 (OpenPose 스타일 지원)"""
    connections = [
        (2, 3), (3, 4),
        (5, 6), (6, 7),
        (1, 2), (1, 5),
        (1, 0),
        (1, 8), (8, 9), (9, 10),
        (1, 11), (11, 12), (12, 13),
        (0, 14), (0, 15),
        (14, 16), (15, 17)
    ]
    
    openpose_colors = [
        (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
        (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
        (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
        (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
        (255, 0, 170), (255, 0, 85)
    ]

    if use_openpose_colors:
        for i, (start_idx, end_idx) in enumerate(connections):
            if len(keypoints) > max(start_idx, end_idx):
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                limb_color = openpose_colors[i % len(openpose_colors)]
                
                x1, y1 = start_point
                x2, y2 = end_point
                mX = (x1 + x2) / 2
                mY = (y1 + y2) / 2
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                angle = np.degrees(np.arctan2(y1 - y2, x1 - x2))
                
                if length > 0:
                    polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), thickness), int(angle), 0, 360, 1)
                    cv2.fillConvexPoly(img, polygon, limb_color)
        
        for i, point in enumerate(keypoints):
            if point[0] > 0 and point[1] > 0:
                cv2.circle(img, point, 4, (255, 255, 255), -1)
    else:
        for start_idx, end_idx in connections:
            if len(keypoints) > max(start_idx, end_idx):
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                cv2.line(img, start_point, end_point, color, thickness)

        for i, point in enumerate(keypoints):
            cv2.circle(img, point, 3, color, -1)


def main():
    original_image_path = "../flask_app/static/uploads/groundimg.jpeg"
    sketch_image_path = "../flask_app/static/sketch/sketch_groundimg.png"
    keypoints_path = "keypoints.json"

    original_img = cv2.imread(original_image_path)
    sketch_img = cv2.imread(sketch_image_path)

    if original_img is None or sketch_img is None:
        print("Error: Could not load images.")
        return

    # 키포인트 로드 및 픽셀 좌표 변환
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

    sketch_pixels = get_sketch_red_pixels(sketch_img)
    if not sketch_pixels:
        print("스케치에서 빨간색 선을 찾지 못했습니다.")
        return

    # 스케치와 가까운 어깨 찾아서 정렬할 팔 결정
    sketch_start_point = min(sketch_pixels, key=lambda p: math.sqrt(
        (p[0] - original_keypoints[2][0]) ** 2 + (p[1] - original_keypoints[2][1]) ** 2))
    arm_to_align = find_closest_arm(sketch_start_point, original_keypoints)
    aligned_keypoints = align_arm_to_sketch(original_keypoints, sketch_pixels, arm_to_align)

    save_keypoints(aligned_keypoints, "keypoints_aligned.json")

    # 결과 이미지 생성 및 저장
    result_img = original_img.copy()
    draw_skeleton_on_image(result_img, original_keypoints, (0, 0, 255), 2)
    draw_skeleton_on_image(result_img, aligned_keypoints, (0, 255, 0), 3, use_openpose_colors=True)
    
    black_bg = np.zeros_like(original_img)
    draw_skeleton_on_image(black_bg, aligned_keypoints, (0, 255, 0), 4, use_openpose_colors=True)
    cv2.imwrite("aligned_pose_only.png", black_bg)
    print("OpenPose 스타일 aligned 포즈가 'aligned_pose_only.png'로 저장되었습니다.")

    cv2.putText(result_img, "Red: Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(result_img, "OpenPose: Aligned", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite("aligned_result.jpg", result_img)
    print("결과가 'aligned_result.jpg'로 저장되었습니다.")

    cv2.imshow("Aligned Skeleton", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()