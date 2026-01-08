from src.controlnet_aux import DWposeDetector
from PIL import Image
import torchvision.transforms as transforms
import torch
import os

def init_dwpose_detector(device):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    det_config = os.path.join(base_dir, 'src/configs/yolox_l_8xb8-300e_coco.py')
    det_ckpt = os.path.join(base_dir, 'ckpts/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth')
    pose_config = os.path.join(base_dir, 'src/configs/dwpose-l_384x288.py')
    pose_ckpt = os.path.join(base_dir, 'ckpts/dw-ll_ucoco_384.pth')

    dwpose_model = DWposeDetector(
        det_config=det_config,
        det_ckpt=det_ckpt,
        pose_config=pose_config,
        pose_ckpt=pose_ckpt,
        device=device
    )
    return dwpose_model.to(device)


def inference_pose(img_path, image_size=(1024, 1024)):
    device = torch.device(f"cuda:{0}")
    model = init_dwpose_detector(device=device)
    pil_image = Image.open(img_path).convert("RGB").resize(image_size, Image.BICUBIC)
    dwpose_image = model(pil_image, output_type='np', image_resolution=image_size[1])
    save_dwpose_image = Image.fromarray(dwpose_image)
    return save_dwpose_image






