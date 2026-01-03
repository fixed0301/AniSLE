from controlnet_aux import OpenposeDetector
from PIL import Image
import numpy as np
import torch

def extract_pose(image):
    """
    이미지에서 포즈 추출 (PIL Image 또는 경로)
    
    Returns:
        pose_image: OpenPose 스켈레톤 이미지 (PIL Image)
        keypoints: List of [x, y] keypoints (18 body points) in pixel coordinates
    """
    detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    # Ensure image is 512x512
    if image.size != (512, 512):
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
    
    # 포즈 이미지 추출
    pose_image = detector(image, output_type='pil', hand_and_face=False)
    
    # Keypoints 추출: detect_poses()로 실제 keypoints 가져오기
    poses = detector.detect_poses(np.array(image))
    
    if len(poses) == 0:
        print("  Warning: No pose detected, using dummy keypoints")
        keypoints = [[256, 50 + i*25] for i in range(18)]
    else:
        # 첫 번째 사람의 body keypoints 사용 (normalized 0-1 좌표)
        body_keypoints = poses[0].body.keypoints
        
        # Pixel coordinates로 변환 (512x512 기준)
        w, h = image.size
        keypoints = []
        for kp in body_keypoints:
            if kp is not None:
                x = int(kp.x * w)
                y = int(kp.y * h)
                keypoints.append([x, y])
            else:
                # None인 경우 (감지 실패) - 이전 keypoint나 중심점 사용
                if len(keypoints) > 0:
                    keypoints.append(keypoints[-1])  # 이전 점 복사
                else:
                    keypoints.append([w//2, h//2])  # 중심점
        
        print(f"  ✓ Extracted {len(body_keypoints)} body keypoints from pose")
    
    return pose_image, keypoints

def extract_keypoints_from_pose_image(pose_arr, image_size=(512, 512)):
    """
    포즈 이미지에서 keypoints 추출 (개선된 버전)
    
    OpenPose body 18 keypoints:
    0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist,
    5: LShoulder, 6: LElbow, 7: LWrist, 8: MidHip,
    9: RHip, 10: RKnee, 11: RAnkle, 12: LHip,
    13: LKnee, 14: LAnkle, 15: REye, 16: LEye,
    17: REar, 18: LEar
    
    (실제로는 18개지만 1인 경우 17:LEar가 마지막)
    """
    h, w = image_size
    
    # 다양한 색상의 관절 찾기 (OpenPose는 다른 색으로 그림)
    # 흰색 또는 밝은 픽셀이 관절/limb
    brightness = pose_arr.mean(axis=2)
    joint_mask = brightness > 100
    
    if not joint_mask.any():
        print("  Warning: No pose detected, using dummy keypoints")
        return [[w//2, 50 + i*25] for i in range(18)]
    
    # Connected components로 관절 찾기
    from scipy import ndimage
    try:
        labeled, num_features = ndimage.label(joint_mask)
        
        if num_features < 5:  # 너무 적으면 실패로 간주
            print(f"  Warning: Only {num_features} features detected, using dummy keypoints")
            return [[w//2, 50 + i*25] for i in range(18)]
        
        # 각 component의 중심점 계산
        centers = []
        for i in range(1, num_features + 1):
            component_mask = (labeled == i)
            y_coords, x_coords = np.where(component_mask)
            if len(x_coords) > 0:
                centers.append([int(np.mean(x_coords)), int(np.mean(y_coords))])
        
        # y좌표로 정렬 (위에서 아래로)
        centers.sort(key=lambda p: p[1])
        
        # 18개로 맞추기
        if len(centers) >= 18:
            keypoints = centers[:18]
        else:
            # 부족한 부분은 보간
            keypoints = centers + [[w//2, h//2]] * (18 - len(centers))
        
        print(f"  ✓ Extracted {len(centers)} pose features -> 18 keypoints")
        return keypoints
        
    except ImportError:
        print("  Warning: scipy not available, using simplified extraction")
        # Fallback: y 좌표 기반 단순 클러스터링
        y_coords, x_coords = np.where(joint_mask)
        sorted_indices = np.argsort(y_coords)
        chunk_size = max(1, len(sorted_indices) // 18)
        
        keypoints = []
        for i in range(18):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(sorted_indices))
            
            if end_idx > start_idx:
                chunk_indices = sorted_indices[start_idx:end_idx]
                x_mean = int(np.mean(x_coords[chunk_indices]))
                y_mean = int(np.mean(y_coords[chunk_indices]))
                keypoints.append([x_mean, y_mean])
            else:
                keypoints.append([w//2, 50 + i*25])
        
        return keypoints
    
    return keypoints

class SimpleOpenPoseExtractor:
    def __init__(self, device="cpu"):
        self.device = device
        self.detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        print(f"✓ OpenPose 초기화 완료 (device: {device})")
    
    def extract_pose(self, image_path):
        """
        이미지에서 포즈 추출
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            pose_image: OpenPose 스켈레톤 이미지 (PIL Image)
            keypoints: List of [x, y] keypoints (18 body points) in pixel coordinates
        """
        image = Image.open(image_path).convert("RGB")
        
        # Ensure image is 512x512
        if image.size != (512, 512):
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
        
        # 포즈 이미지 생성
        pose_image = self.detector(image, output_type='pil', hand_and_face=False)
        
        # Keypoints 추출
        poses = self.detector.detect_poses(np.array(image))
        
        if len(poses) == 0:
            print("  Warning: No pose detected")
            keypoints = [[256, 50 + i*25] for i in range(18)]
        else:
            body_keypoints = poses[0].body.keypoints
            w, h = image.size
            keypoints = []
            for kp in body_keypoints:
                if kp is not None:
                    keypoints.append([int(kp.x * w), int(kp.y * h)])
                else:
                    keypoints.append(keypoints[-1] if keypoints else [w//2, h//2])
        
        return pose_image, keypoints
    
    def process_sketch_to_pose(self, sketch_path):
        """
        스케치를 포즈 컨디션 이미지로 변환
        
        Args:
            sketch_path: 스케치 파일 경로
            
        Returns:
            pose_condition: 포즈 컨디션 이미지 (numpy array, RGB)
        """
        sketch = Image.open(sketch_path).convert("RGBA")
        sketch_arr = np.array(sketch)
        
        # 빨간색 스케치를 흰색 선으로 변환
        pose_condition = np.zeros((sketch_arr.shape[0], sketch_arr.shape[1], 3), dtype=np.uint8)
        
        # 알파 채널이 있으면 사용
        if sketch_arr.shape[2] == 4:
            alpha = sketch_arr[:, :, 3]
            mask = alpha > 50
        else:
            # RGB - 빨간색 또는 어두운 픽셀
            red_mask = (sketch_arr[:, :, 0] > 150) & (sketch_arr[:, :, 1] < 100) & (sketch_arr[:, :, 2] < 100)
            dark_mask = np.mean(sketch_arr[:, :, :3], axis=2) < 200
            mask = red_mask | dark_mask
        
        # 흰색으로 변환
        pose_condition[mask] = [255, 255, 255]
        
        return pose_condition


if __name__ == "__main__":
    # 테스트
    extractor = SimpleOpenPoseExtractor()
    
    test_image = "/workspace/AniSLE/flask_app/static/uploads/0.jpeg"
    import os
    if os.path.exists(test_image):
        pose = extractor.extract_pose(test_image)
        pose.save("/workspace/AniSLE/test_openpose.png")
        print(f"✓ 포즈 추출 완료: test_openpose.png")
    else:
        print(f"❌ 이미지 파일 없음: {test_image}")
