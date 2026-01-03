import json
from PIL import Image
from easy_dwpose import DWposeDetector


device = "cuda:0"
detector = DWposeDetector(device=device)

def create_json(idx, save_dwpose_image=True):
    input_image = Image.open("../flask_app/static/uploads/{idx}.jpeg").convert("RGB")
    pose = detector(input_image,draw_pose=None)

    if save_dwpose_image == True:
        skeleton = detector(input_image, output_type="pil", include_hands=True, include_face=True)
        skeleton.save("pose_img_{idx}.png")

    with open("pose_keypoints_{idx}.json", "w") as f:
        json.dump({
            "bodies": pose['bodies'].tolist(),
            "body_scores": pose['body_scores'].tolist(),
            "hands": pose['hands'].tolist(),
            "hands_scores": pose['hands_scores'].tolist(),
            "faces": pose['faces'].tolist(),
            "faces_scores": pose['faces_scores'].tolist()
        }, f, indent=4)

