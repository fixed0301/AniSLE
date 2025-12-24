"""
ControlNet OpenPose Detector 사용
"""
from controlnet_aux import OpenposeDetector
from PIL import Image
import numpy as np
import torch

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
        """
        image = Image.open(image_path).convert("RGB")
        pose_image = self.detector(image)
        return pose_image
    
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
