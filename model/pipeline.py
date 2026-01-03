"""
통합 파이프라인: 포즈 추출 → 정렬 → 마스크 생성 → 이미지 생성
파일 I/O 최소화, 메모리에서 데이터 직접 전달
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PCDMs'))

import cv2
import numpy as np
import json
from PIL import Image
import torch
from pathlib import Path


def extract_pose_dwpose(image_path):
    """DWPose로 포즈 추출 - keypoints 반환"""
    from controlnet_aux import DWposeDetector
    import cv2
    
    print("Initializing DWPose detector...")
    detector = DWposeDetector()
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")
    
    # DWPose 실행 - pose 이미지 반환
    print("Detecting pose...")
    result = detector(image, output_type='np')
    
    # controlnet_aux DWposeDetector는 pose 이미지만 반환하므로
    # keypoints를 직접 추출할 수 없음
    # 대신 기존 파일이 있으면 사용
    h, w = image.shape[:2]
    
    # 임시: 간단한 더미 keypoints (18개)
    # 실제로는 pose 이미지에서 skeleton parsing 필요
    print("Warning: Using dummy keypoints. Implement proper keypoint extraction.")
    dummy_keypoints = [[0.5, 0.3 + i*0.03] for i in range(18)]
    
    return dummy_keypoints


def visualize_aligned_skeleton(keypoints, sketch_img, output_path):
    """Aligned skeleton을 시각화해서 저장 (draw_skeleton.py 방식)"""
    from PIL import Image, ImageDraw
    
    h, w = sketch_img.shape[:2]
    
    # 검은 배경 PIL 이미지 생성
    vis_img = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(vis_img)
    
    print(f"  Visualizing skeleton: image size {w}x{h}")
    print(f"  Keypoints range: x=[{min([k[0] for k in keypoints]):.1f}, {max([k[0] for k in keypoints]):.1f}], y=[{min([k[1] for k in keypoints]):.1f}, {max([k[1] for k in keypoints]):.1f}]")
    
    # Keypoints are already in pixel coordinates
    kpts = [(int(kpt[0]), int(kpt[1])) for kpt in keypoints]
    
    # OpenPose body connections (draw_skeleton.py와 동일)
    connections = [
        (2, 3), (3, 4),  # Right arm
        (5, 6), (6, 7),  # Left arm
        (1, 2), (1, 5),  # Neck to shoulders
        (1, 0),  # Neck to nose
        (1, 8), (8, 9), (9, 10),  # Torso and right hip/leg
        (1, 11), (11, 12), (12, 13),  # Torso and left hip/leg (수정: 8->1)
        (0, 14), (0, 15),  # Nose to eyes
        (14, 16), (15, 17)  # Eyes to ears
    ]
    
    # 관절 연결선 그리기 (녹색)
    for start_idx, end_idx in connections:
        if start_idx < len(kpts) and end_idx < len(kpts):
            draw.line([kpts[start_idx], kpts[end_idx]], fill=(0, 255, 0), width=3)
    
    # 관절 점 그리기 (파란색)
    for point in kpts:
        radius = 5
        draw.ellipse(
            [(point[0] - radius, point[1] - radius), 
             (point[0] + radius, point[1] + radius)], 
            fill=(0, 0, 255)
        )
    
    # 스케치 빨간색 선도 오버레이
    sketch_np = np.array(sketch_img)
    lower_red = np.array([0, 0, 100])
    upper_red = np.array([50, 50, 255])
    mask_red = cv2.inRange(sketch_np, lower_red, upper_red)
    
    # 빨간색 픽셀을 vis_img에 그리기
    red_pixels = np.where(mask_red > 0)
    for y, x in zip(red_pixels[0], red_pixels[1]):
        draw.point((x, y), fill=(255, 0, 0))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    vis_img.save(output_path)
    print(f"  ✓ Saved aligned skeleton visualization to {output_path}")
    print(f"  Saved visualization to {output_path}")


def generate_mask_from_keypoints(original_kpts, aligned_kpts, image_shape, threshold=30):
    """두 keypoints 비교해서 마스크 생성"""
    import tempfile
    
    h, w = image_shape[:2]
    
    # Keypoints are already in pixel coordinates, no need to denormalize
    original_kpts_pixel = [[float(kpt[0]), float(kpt[1])] for kpt in original_kpts]
    aligned_kpts_pixel = [[float(kpt[0]), float(kpt[1])] for kpt in aligned_kpts]
    
    # Save keypoints as temporary JSON files (in pixel coordinates)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f1:
        import json
        json.dump(original_kpts_pixel, f1)
        source_path = f1.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f2:
        json.dump(aligned_kpts_pixel, f2)
        target_path = f2.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.png', delete=False) as f3:
        mask_path = f3.name
    
    try:
        from mask.generate_mask_from_keypoints import generate_mask_from_keypoint_diff
        
        # Generate mask using existing function
        mask_pil, moved_indices, distances = generate_mask_from_keypoint_diff(
            source_kp_path=source_path,
            target_kp_path=target_path,
            output_path=mask_path,
            image_size=(w, h),
            threshold=threshold,
            joint_radius=50,  # 관절 주변 반경 더 증가 (40 -> 50)
            limb_thickness=55  # 관절 연결선 두께 더 증가 (30 -> 45)
        )
        
        # Convert PIL to numpy
        mask = np.array(mask_pil)
        return mask
    
    finally:
        # Cleanup temp files
        import os
        for path in [source_path, target_path, mask_path]:
            if os.path.exists(path):
                os.unlink(path)


def run_full_pipeline(idx, flask_app_dir):
    """
    전체 파이프라인 실행
    
    Args:
        idx: 이미지 인덱스
        flask_app_dir: Flask 앱 디렉토리 경로
    """
    print(f"\n=== 파이프라인 시작: idx={idx} ===")
    
    # 경로 설정
    model_dir = os.path.dirname(__file__)
    
    # Try multiple extensions for source image
    upload_dir = os.path.join(flask_app_dir, 'data', 'uploads')
    image_path = None
    for ext in ['.png', '.jpeg', '.jpg', '.PNG', '.JPEG', '.JPG']:
        test_path = os.path.join(upload_dir, f'{idx}{ext}')
        if os.path.exists(test_path):
            image_path = test_path
            break
    
    if image_path is None:
        raise FileNotFoundError(f"원본 이미지를 찾을 수 없습니다: {upload_dir}/{idx}.*")
    
    sketch_path = os.path.join(flask_app_dir, 'data', 'sketch', f'sketch_{idx}.png')
    keypoints_path = os.path.join(model_dir, 'pose', 'pose_json', f'keypoints_{idx}.json')
    
    print(f"✓ Image path: {image_path}")
    
    # 1. 포즈 keypoints 로드 (이미 추출되어 있음)
    print("Step 1/4: 포즈 keypoints 로드 중...")
    if not os.path.exists(keypoints_path):
        raise FileNotFoundError(f"키포인트 파일이 없습니다: {keypoints_path}")
    
    with open(keypoints_path, 'r') as f:
        original_keypoints = json.load(f)
    print(f"✓ 포즈 로드 완료 ({len(original_keypoints)} keypoints)")
    
    # 2. 스케치 로드 및 정렬
    print("Step 2/4: 포즈 정렬 중...")
    sketch_img = cv2.imread(sketch_path)
    if sketch_img is None:
        raise FileNotFoundError(f"스케치 파일을 읽을 수 없습니다: {sketch_path}")
    
    # align.py의 align 함수 호출
    from pose.utils.align import align
    align(idx)
    
    # 정렬된 keypoints 로드
    aligned_kpts_path = os.path.join(model_dir, 'pose', 'pose_json', f'keypoints_aligned_{idx}.json')
    with open(aligned_kpts_path, 'r') as f:
        aligned_keypoints = json.load(f)
    
    print(f"✓ 포즈 정렬 완료")
    print(f"  Original keypoints sample: {original_keypoints[:3]}")
    print(f"  Aligned keypoints sample: {aligned_keypoints[:3]}")
    
    # Aligned skeleton 시각화 저장
    aligned_pose_img_path = os.path.join(model_dir, 'pose', 'pose_img', f'aligned_{idx}.png')
    visualize_aligned_skeleton(aligned_keypoints, sketch_img, aligned_pose_img_path)
    print(f"✓ 정렬된 포즈 시각화 저장: {aligned_pose_img_path}")
    
    # 3. 마스크 생성
    print("Step 3/4: 마스크 생성 중...")
    # 512x512 크기로 마스크 생성 (업로드된 이미지 크기와 동일)
    mask = generate_mask_from_keypoints(
        original_keypoints, 
        aligned_keypoints,
        (512, 512, 3),  # 고정 크기
        threshold=5  # 임계값 더 낮춤 (15 -> 5) - 매우 작은 변화도 감지
    )
    
    # 마스크 저장 (PCDMs가 파일로 읽어야 하므로)
    mask_path = os.path.join(model_dir, 'mask', 'mask_img', f'mask_{idx}.png')
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    cv2.imwrite(mask_path, mask)
    print(f"✓ 마스크 생성 완료: {mask_path}")
    
    # Aligned keypoints도 저장 (PCDMs가 사용)
    aligned_kpts_path = os.path.join(model_dir, 'pose', 'pose_json', f'keypoints_aligned_{idx}.json')
    os.makedirs(os.path.dirname(aligned_kpts_path), exist_ok=True)
    with open(aligned_kpts_path, 'w') as f:
        json.dump(aligned_keypoints, f, indent=2)
    print(f"✓ 정렬된 키포인트 저장: {aligned_kpts_path}")
    
    # 4. PCDMs 이미지 생성
    print("Step 4/4: 이미지 생성 중 (약 2-3분 소요)...")
    from generate.simple_demo import run_inference
    
    # 결과 저장 경로
    result_path = os.path.join(flask_app_dir, 'data', 'results', f'result_{idx}.png')
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    result_path = run_inference(
        source_image_path=image_path,
        target_pose_json=aligned_kpts_path,
        mask_path=mask_path,
        output_path=result_path,
        num_steps=75,
        guidance_scale=7.0
    )
    print(f"✓ 이미지 생성 완료: {result_path}")
    
    print(f"\n=== 파이프라인 완료: idx={idx} ===\n")
    return result_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Full pipeline: pose extraction → alignment → mask → generation')
    parser.add_argument('--idx', type=str, required=True, help='Image index')
    args = parser.parse_args()
    
    # Flask 앱 디렉토리 찾기
    script_dir = Path(__file__).parent
    flask_app_dir = script_dir.parent / 'flask_app'
    
    try:
        result = run_full_pipeline(args.idx, str(flask_app_dir))
        print(f"SUCCESS: {result}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
