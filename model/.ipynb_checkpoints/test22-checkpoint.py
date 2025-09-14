import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from segment_anything import SamPredictor, sam_model_registry

def load_image(path: str) -> np.ndarray:
    """이미지 로드 (RGB 형식으로)"""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_pose_json(path: str) -> dict:
    """포즈 JSON 파일 로드"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def init_sam(model_type: str = "vit_h", checkpoint: str = None, device="cuda"):
    """SAM 모델 초기화"""
    if checkpoint is None:
        raise ValueError("Please provide a SAM checkpoint path.")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def extract_keypoints_from_dict(pose_dict: dict, image_size: Tuple[int,int]) -> List[Tuple[float, float, float]]:
    """다양한 포즈 JSON 형식에서 키포인트 추출"""
    H, W = image_size

    # DWpose style: {"bodies": [[[x,y],...]]} OR {"bodies": [[x,y], ...]}
    if 'bodies' in pose_dict and len(pose_dict['bodies']) > 0:
        arr = pose_dict['bodies'][0]
        # case 1: each element is [x,y]
        if all(isinstance(pt, list) and len(pt) == 2 for pt in arr):
            kps = [(pt[0], pt[1], 1.0) for pt in arr]
            return kps
        # case 2: flat list [x1,y1, x2,y2, ...]
        if all(isinstance(val, (int,float)) for val in arr):
            kps = [(arr[i], arr[i+1], 1.0) for i in range(0, len(arr), 2)]
            return kps

    # Aligned pose: [[x,y], [x,y], ...]
    if isinstance(pose_dict, list) and all(isinstance(pt, list) and len(pt)==2 for pt in pose_dict):
        return [(x/W, y/H, 1.0) for x,y in pose_dict]

    # direct x,y arrays
    if 'x' in pose_dict and 'y' in pose_dict:
        xs = pose_dict['x']; ys = pose_dict['y']; vs = pose_dict.get('v', [1.0]*len(xs))
        return list(zip(xs, ys, vs))

    # fallback - 다른 형식 시도
    for v in pose_dict.values():
        if isinstance(v, list) and len(v) % 3 == 0 and len(v) >= 3:
            arr = v
            return [(arr[i], arr[i+1], arr[i+2]) for i in range(0, len(arr), 3)]
    return []

def get_pose_difference_points(orig_kps: List[Tuple[float, float, float]], 
                              aligned_kps: List[Tuple[float, float, float]], 
                              image_size: Tuple[int,int], 
                              thresh: float = 0.05) -> List[Tuple[int, int]]:
    """포즈 차이가 있는 키포인트들의 픽셀 좌표 반환"""
    H, W = image_size
    diff_points = []

    for (x0,y0,v0), (x1,y1,v1) in zip(orig_kps, aligned_kps):
        if v0 < 0.3 or v1 < 0.3:  # visibility threshold
            continue
        
        # 정규화된 좌표인지 확인하고 처리
        if x0 <= 1.0 and y0 <= 1.0:  # normalized coordinates
            x0, y0 = x0 * W, y0 * H
        if x1 <= 1.0 and y1 <= 1.0:  # normalized coordinates
            x1, y1 = x1 * W, y1 * H
            
        dx, dy = abs(x0-x1), abs(y0-y1)
        if dx > thresh * W or dy > thresh * H:  # threshold 조정
            cx, cy = int(x1), int(y1)
            # 이미지 경계 내에서만 추가
            if 0 <= cx < W and 0 <= cy < H:
                diff_points.append((cx, cy))
    
    return diff_points

def get_sam_masks_from_points(predictor, image: np.ndarray, points: List[Tuple[int, int]]) -> List[np.ndarray]:
    """SAM을 사용해 포인트들로부터 마스크 생성"""
    if len(points) == 0:
        return []
    
    # 이미지를 SAM에 설정
    predictor.set_image(image)
    
    # 포인트들을 numpy 배열로 변환
    input_points = np.array(points)
    input_labels = np.ones(len(points))  # 모든 포인트를 positive로
    
    # SAM으로 마스크 예측
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )
    
    # 가장 높은 점수의 마스크 선택
    best_mask_idx = np.argmax(scores)
    best_mask = masks[best_mask_idx]
    
    return best_mask

def get_sam_masks_grid(predictor, image: np.ndarray, grid_size: int = 32) -> List[np.ndarray]:
    """그리드 방식으로 SAM 마스크들 생성 (원본 코드)"""
    image_torch = image.astype(np.uint8)
    predictor.set_image(image_torch)

    h, w = image.shape[:2]
    masks = []

    ys = np.linspace(0, h - 1, grid_size, dtype=int)
    xs = np.linspace(0, w - 1, grid_size, dtype=int)

    points = [[x, y] for y in ys for x in xs]
    points = np.array(points)

    chunk = 512
    for i in range(0, len(points), chunk):
        pts = points[i:i + chunk]
        input_points = pts
        input_labels = np.ones(len(pts), dtype=int)
        masks_out, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        for mset in masks_out:
            for m in mset:
                mask = m.astype(bool)
                if mask.sum() < 0.001 * h * w:
                    continue
                masks.append(mask)

    # 중복 제거
    unique = []
    seen = set()
    for m in masks:
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        bb = (min(xs), min(ys), max(xs), max(ys))
        if bb in seen:
            continue
        seen.add(bb)
        unique.append(m)
    return unique

def select_best_mask_for_pose(masks: List[np.ndarray], diff_points: List[Tuple[int, int]]) -> np.ndarray:
    """포즈 차이 포인트들과 가장 잘 맞는 마스크 선택"""
    if len(masks) == 0:
        return np.zeros((100, 100), dtype=bool)
    
    if len(diff_points) == 0:
        return max(masks, key=lambda x: x.sum())
    
    h, w = masks[0].shape
    pose_mask = np.zeros((h, w), dtype=bool)
    
    # 포즈 차이 포인트들 주변에 작은 마스크 생성
    for x, y in diff_points:
        if 0 <= y < h and 0 <= x < w:
            y1, y2 = max(0, y-10), min(h, y+10)
            x1, x2 = max(0, x-10), min(w, x+10)
            pose_mask[y1:y2, x1:x2] = True

    best_mask, best_iou = None, 0.0
    for m in masks:
        inter = np.logical_and(m, pose_mask).sum()
        union = np.logical_or(m, pose_mask).sum()
        if union == 0:
            continue
        iou = inter / union
        if iou > best_iou:
            best_iou, best_mask = iou, m
    
    if best_mask is None:
        best_mask = max(masks, key=lambda x: x.sum())
    
    return best_mask

def visualize_sam_results(image: np.ndarray, 
                         orig_kps: List[Tuple[float, float, float]], 
                         aligned_kps: List[Tuple[float, float, float]], 
                         diff_points: List[Tuple[int, int]],
                         sam_mask: np.ndarray):
    """SAM 결과를 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    H, W = image.shape[:2]
    
    # 원본 이미지 + 원본 포즈
    axes[0,0].imshow(image)
    for i, (x, y, v) in enumerate(orig_kps):
        if v > 0.3:
            if x <= 1.0 and y <= 1.0:
                x, y = x * W, y * H
            axes[0,0].plot(x, y, 'ro', markersize=6)
    axes[0,0].set_title('Original Image + Original Pose')
    axes[0,0].axis('off')
    
    # 정렬된 포즈 + 차이 포인트들
    axes[0,1].imshow(image)
    for i, (x, y, v) in enumerate(aligned_kps):
        if v > 0.3:
            if x <= 1.0 and y <= 1.0:
                x, y = x * W, y * H
            axes[0,1].plot(x, y, 'bo', markersize=6)
    
    # 차이 포인트들 강조
    for x, y in diff_points:
        axes[0,1].plot(x, y, 'r*', markersize=12)
    axes[0,1].set_title('Aligned Pose + Difference Points')
    axes[0,1].axis('off')
    
    # SAM 마스크
    axes[1,0].imshow(sam_mask, cmap='gray')
    axes[1,0].set_title('SAM Generated Mask')
    axes[1,0].axis('off')
    
    # 원본 이미지 + 마스크 오버레이
    masked_image = image.copy()
    masked_image[sam_mask] = masked_image[sam_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
    axes[1,1].imshow(masked_image.astype(np.uint8))
    axes[1,1].set_title('Image + SAM Mask Overlay')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()

def main_sam_mask_generation(image_path: str, 
                           orig_pose_json: str, 
                           aligned_pose_json: str, 
                           sam_checkpoint: str,
                           thresh: float = 0.05,
                           use_pose_points: bool = True):
    """메인 함수: SAM을 사용한 마스크 생성과 시각화"""
    
    # 이미지 로드
    try:
        img_np = load_image(image_path)
        H, W = img_np.shape[:2]
        print(f"Image loaded: {W}x{H}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # SAM 초기화
    try:
        sam_predictor = init_sam(model_type='vit_h', checkpoint=sam_checkpoint, device='cuda')
        print("SAM model initialized")
    except Exception as e:
        print(f"Error initializing SAM: {e}")
        return
    
    # 포즈 JSON 로드
    try:
        orig_pose = load_pose_json(orig_pose_json)
        aligned_pose = load_pose_json(aligned_pose_json)
        print("Pose JSONs loaded successfully")
    except Exception as e:
        print(f"Error loading pose JSONs: {e}")
        return
    
    # 키포인트 추출
    try:
        kps_orig = extract_keypoints_from_dict(orig_pose, (H, W))
        kps_aligned = extract_keypoints_from_dict(aligned_pose, (H, W))
        print(f"Original keypoints: {len(kps_orig)}")
        print(f"Aligned keypoints: {len(kps_aligned)}")
    except Exception as e:
        print(f"Error extracting keypoints: {e}")
        return
    
    # 포즈 차이 포인트 추출
    try:
        diff_points = get_pose_difference_points(kps_orig, kps_aligned, (H, W), thresh)
        print(f"Pose difference points: {len(diff_points)}")
    except Exception as e:
        print(f"Error getting pose difference points: {e}")
        return
    
    # SAM 마스크 생성
    try:
        if use_pose_points and len(diff_points) > 0:
            # 포즈 차이 포인트들로부터 직접 마스크 생성
            sam_mask = get_sam_masks_from_points(sam_predictor, img_np, diff_points)
            print("SAM mask generated from pose difference points")
        else:
            # 그리드 방식으로 마스크들 생성 후 최적 선택
            all_masks = get_sam_masks_grid(sam_predictor, img_np, grid_size=32)
            sam_mask = select_best_mask_for_pose(all_masks, diff_points)
            print(f"SAM mask selected from {len(all_masks)} grid masks")
            
        print(f"Final mask coverage: {np.count_nonzero(sam_mask)} pixels")
    except Exception as e:
        print(f"Error generating SAM mask: {e}")
        return
    
    # 시각화
    try:
        visualize_sam_results(img_np, kps_orig, kps_aligned, diff_points, sam_mask)
    except Exception as e:
        print(f"Error in visualization: {e}")
        return
    
    # 마스크 저장
    try:
        mask_path = image_path.replace('.', '_sam_mask.')
        cv2.imwrite(mask_path, (sam_mask.astype(np.uint8) * 255))
        print(f"SAM mask saved to: {mask_path}")
    except Exception as e:
        print(f"Error saving mask: {e}")

# 사용 예시
if __name__ == "__main__":
    # 파일 경로들을 실제 경로로 변경하세요
    image_path = "../flask_app/static/uploads/groundimg.jpeg"
    orig_pose_json = "keypoints.json"
    aligned_pose_json = "keypoints_aligned.json"
    sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
    
    # SAM 마스크 생성 및 시각화 실행
    main_sam_mask_generation(image_path, orig_pose_json, aligned_pose_json, sam_checkpoint,
                           thresh=0.05, use_pose_points=True)