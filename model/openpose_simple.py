from controlnet_aux import OpenposeDetector
from PIL import Image
import numpy as np
import torch

def extract_pose(image):
    """이미지에서 OpenPose 스켈레톤과 keypoints 추출"""
    detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    if image.size != (512, 512):
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
    
    pose_image = detector(image, output_type='pil', hand_and_face=False)
    poses = detector.detect_poses(np.array(image))
    
    if len(poses) == 0:
        print("  Warning: No pose detected, using dummy keypoints")
        keypoints = [[256, 50 + i*25] for i in range(18)]
    else:
        body_keypoints = poses[0].body.keypoints
        w, h = image.size
        keypoints = []
        for kp in body_keypoints:
            if kp is not None:
                x = int(kp.x * w)
                y = int(kp.y * h)
                keypoints.append([x, y])
            else:
                if len(keypoints) > 0:
                    keypoints.append(keypoints[-1])
                else:
                    keypoints.append([w//2, h//2])
        print(f"  ✓ Extracted {len(body_keypoints)} body keypoints from pose")
    
    return pose_image, keypoints

def extract_keypoints_from_pose_image(pose_arr, image_size=(512, 512)):
    """포즈 이미지에서 18개 body keypoints 추출 (Connected components 방식)"""
    h, w = image_size
    
    brightness = pose_arr.mean(axis=2)
    joint_mask = brightness > 100
    
    if not joint_mask.any():
        print("  Warning: No pose detected, using dummy keypoints")
        return [[w//2, 50 + i*25] for i in range(18)]
    
    from scipy import ndimage
    try:
        labeled, num_features = ndimage.label(joint_mask)
        
        if num_features < 5:
            print(f"  Warning: Only {num_features} features detected, using dummy keypoints")
            return [[w//2, 50 + i*25] for i in range(18)]
        
        centers = []
        for i in range(1, num_features + 1):
            component_mask = (labeled == i)
            y_coords, x_coords = np.where(component_mask)
            if len(x_coords) > 0:
                centers.append([int(np.mean(x_coords)), int(np.mean(y_coords))])
        
        centers.sort(key=lambda p: p[1])
        
        if len(centers) >= 18:
            keypoints = centers[:18]
        else:
            keypoints = centers + [[w//2, h//2]] * (18 - len(centers))
        
        print(f"  ✓ Extracted {len(centers)} pose features -> 18 keypoints")
        return keypoints
        
    except ImportError:
        print("  Warning: scipy not available, using simplified extraction")
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
        """이미지에서 OpenPose 스켈레톤과 18개 body keypoints 추출"""
        image = Image.open(image_path).convert("RGB")
        
        if image.size != (512, 512):
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
        
        pose_image = self.detector(image, output_type='pil', hand_and_face=False)
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
        """스케치를 포즈 컨디션 이미지로 변환 (빨간색 스케치 → 흰색 선)"""
        sketch = Image.open(sketch_path).convert("RGBA")
        sketch_arr = np.array(sketch)
        pose_condition = np.zeros((sketch_arr.shape[0], sketch_arr.shape[1], 3), dtype=np.uint8)
        
        if sketch_arr.shape[2] == 4:
            alpha = sketch_arr[:, :, 3]
            mask = alpha > 50
        else:
            red_mask = (sketch_arr[:, :, 0] > 150) & (sketch_arr[:, :, 1] < 100) & (sketch_arr[:, :, 2] < 100)
            dark_mask = np.mean(sketch_arr[:, :, :3], axis=2) < 200
            mask = red_mask | dark_mask
        
        pose_condition[mask] = [255, 255, 255]
        return pose_condition


if __name__ == "__main__":
    extractor = SimpleOpenPoseExtractor()
    test_image = "/workspace/AniSLE/flask_app/static/uploads/0.jpeg"
    import os
    if os.path.exists(test_image):
        pose = extractor.extract_pose(test_image)
        pose.save("/workspace/AniSLE/test_openpose.png")
        print(f"✓ 포즈 추출 완료: test_openpose.png")
    else:
        print(f"❌ 이미지 파일 없음: {test_image}")
