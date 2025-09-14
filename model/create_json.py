from PIL import Image
from easy_dwpose import DWposeDetector

device = "cpu"
detector = DWposeDetector(device=device)
input_image = Image.open("../flask_app/static/uploads/groundimg.jpeg").convert("RGB")

pose = detector(input_image,draw_pose=None)

# 이미지 저장
# skeleton = detector(input_image, output_type="pil", include_hands=True, include_face=True)
# skeleton.save("skeleton.png")

print("Body Keypoints (x, y, confidence):")
for i, kp in enumerate(pose['bodies']):
    if pose['body_scores'][i // 18][i % 18] != -1:  # 유효한 키포인트만 출력
        print(f"  Point {i}: x={kp[0]}, y={kp[1]}, confidence={pose['body_scores'][i // 18][i % 18]}")

print("\nHand Keypoints (x, y, confidence):")
for i, kp in enumerate(pose['hands']):
    if pose['hands_scores'][i // 21][i % 21] > 0:  # 신뢰도 0 초과인 경우
        print(f"  Point {i}: x={kp[0]}, y={kp[1]}, confidence={pose['hands_scores'][i // 21][i % 21]}")

print("\nFace Keypoints (x, y, confidence):")
for i, kp in enumerate(pose['faces']):
    if pose['faces_scores'][i // 68][i % 68] > 0:
        print(f"  Point {i}: x={kp[0]}, y={kp[1]}, confidence={pose['faces_scores'][i // 68][i % 68]}")


import json
with open("keypoints.json", "w") as f:
    json.dump({
        "bodies": pose['bodies'].tolist(),
        "body_scores": pose['body_scores'].tolist(),
        "hands": pose['hands'].tolist(),
        "hands_scores": pose['hands_scores'].tolist(),
        "faces": pose['faces'].tolist(),
        "faces_scores": pose['faces_scores'].tolist()
    }, f, indent=4)

