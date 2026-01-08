from flask import Flask, request, render_template, send_from_directory, abort, Response, jsonify, redirect, url_for
from PIL import Image
from werkzeug.utils import secure_filename
import os
import glob
import sys
import subprocess
import json
import time
import threading
import numpy as np
import cv2

model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
sys.path.insert(0, model_dir)

app = Flask(__name__)

processing_status = {}

APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_DIR, 'data', 'uploads')
SKETCH_FOLDER = os.path.join(APP_DIR, 'data', 'sketch')
RESULT_FOLDER = os.path.join(APP_DIR, 'data', 'results')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SKETCH_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SKETCH_FOLDER'] = SKETCH_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def get_next_index():
    files = glob.glob(os.path.join(UPLOAD_FOLDER, '*.*'))
    indices = []
    for f in files:
        basename = os.path.basename(f)
        name = os.path.splitext(basename)[0]
        if name.isdigit():
            indices.append(int(name))
    return max(indices) + 1 if indices else 0

@app.route('/health')
def health_check():
    return {'status': 'ok', 'message': 'Flask server is running'}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename:
                index = get_next_index()
                ext = os.path.splitext(file.filename)[1]
                filename = f"{index}{ext}"
                
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image = Image.open(file).convert('RGB').resize((1024, 1024), Image.Resampling.LANCZOS)
                image.save(file_path)
                
                try:
                    print(f"Extracting pose for {filename}...")
                    from openpose_simple import extract_pose
                    
                    pose_img, keypoints = extract_pose(image)
                    
                    pose_img_path = os.path.join(model_dir, 'pose', 'pose_img', f'{index}.png')
                    os.makedirs(os.path.dirname(pose_img_path), exist_ok=True)
                    pose_img.save(pose_img_path)
                    
                    keypoints_path = os.path.join(model_dir, 'pose', 'pose_json', f'keypoints_{index}.json')
                    os.makedirs(os.path.dirname(keypoints_path), exist_ok=True)
                    with open(keypoints_path, 'w') as f:
                        json.dump({'keypoints': keypoints}, f, indent=2)
                    
                    print(f"✓ Pose extracted: {len(keypoints)} keypoints")
                except Exception as e:
                    print(f"Warning: Pose extraction failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                return redirect(url_for('show_image', filename=filename))
    return render_template('index.html', image_path=None)

@app.route('/image/<filename>')
def show_image(filename):
    base = os.path.splitext(filename)[0]
    result_filename = f"result_{base}.png"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    
    if os.path.exists(result_path):
        return render_template('index.html', image_path=filename, result_path=result_filename)
    else:
        return render_template('index.html', image_path=filename)

@app.route('/save_sketch', methods=['POST'])
def save_sketch():
    if 'sketch' in request.files and 'filename' in request.form:
        file = request.files['sketch']
        sketch = Image.open(file.stream).convert('RGBA').resize((1024, 1024), Image.Resampling.LANCZOS)

        orig_filename = secure_filename(request.form['filename'])
        base, _ = os.path.splitext(orig_filename)
        sketch_filename = f"sketch_{base}.png"
        sketch_path = os.path.join(app.config['SKETCH_FOLDER'], sketch_filename)
        sketch.save(sketch_path)

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], orig_filename)
        if not os.path.exists(image_path):
            abort(404, description="Original image not found.")
        image = Image.open(image_path).convert('RGB')

        idx = base
        processing_status[idx] = {'status': 'queued', 'step': 0, 'message': '처리 대기 중...'}
        
        def process_in_background():
            try:
                print(f"Running pipeline for idx={idx}")
                
                processing_status[idx] = {'status': 'processing', 'step': 1, 'message': 'Step 1/4: 포즈 keypoints 추출 중...'}
                
                import openpose_simple
                import align
                from PIL import Image as PILImage
                
                source_img = PILImage.open(image_path).convert('RGB').resize((1024, 1024), PILImage.Resampling.LANCZOS)
                pose_image, keypoints = openpose_simple.extract_pose(source_img)
                
                pose_dir = os.path.join(model_dir, 'pose', 'pose_img')
                pose_json_dir = os.path.join(model_dir, 'pose', 'pose_json')
                os.makedirs(pose_dir, exist_ok=True)
                os.makedirs(pose_json_dir, exist_ok=True)
                
                pose_image.save(os.path.join(pose_dir, f'pose_{idx}.png'))
                
                keypoints_data = {'keypoints': keypoints}
                with open(os.path.join(pose_json_dir, f'keypoints_{idx}.json'), 'w') as f:
                    json.dump(keypoints_data, f)
                
                print(f"  ✓ Extracted {len(keypoints)} keypoints")
                
                processing_status[idx] = {'status': 'processing', 'step': 2, 'message': 'Step 2/4: 팔 keypoints 정렬 중...'}
                
                sketch_img = PILImage.open(sketch_path).convert('RGBA')
                sketch_np = np.array(sketch_img)
                sketch_bgr = cv2.cvtColor(sketch_np, cv2.COLOR_RGBA2BGR)
                
                sketch_pixels = align.get_sketch_red_pixels(sketch_bgr)
                
                if sketch_pixels:
                    sketch_start = min(sketch_pixels, key=lambda p: np.sqrt((p[0] - keypoints[2][0])**2 + (p[1] - keypoints[2][1])**2))
                    arm_to_align = align.find_closest_arm(sketch_start, keypoints)
                    aligned_keypoints = align.align_arm_to_sketch(keypoints, sketch_pixels, arm_to_align)
                else:
                    print("  Warning: No sketch pixels found")
                    aligned_keypoints = keypoints
                    arm_to_align = 'right'
                
                aligned_json_path = os.path.join(pose_json_dir, f'keypoints_aligned_{idx}.json')
                with open(aligned_json_path, 'w') as f:
                    json.dump({'keypoints': aligned_keypoints}, f)
                
                aligned_pose_img = np.array(source_img).copy()
                align.draw_skeleton_on_image(aligned_pose_img, aligned_keypoints, use_openpose_colors=True)
                PILImage.fromarray(aligned_pose_img).save(os.path.join(pose_dir, f'aligned_{idx}.png'))
                
                print(f"  ✓ Aligned pose")
                
                processing_status[idx] = {'status': 'processing', 'step': 3, 'message': 'Step 3/4: 마스크 생성 중...'}
                
                mask_dir = os.path.join(model_dir, 'mask', 'mask_img')
                os.makedirs(mask_dir, exist_ok=True)
                
                mask_array = np.zeros((1024, 1024), dtype=np.uint8)
                
                if arm_to_align == 'right':
                    orig_arm = [keypoints[2], keypoints[3], keypoints[4]]
                    aligned_arm = [aligned_keypoints[2], aligned_keypoints[3], aligned_keypoints[4]]
                elif arm_to_align == 'left':
                    orig_arm = [keypoints[5], keypoints[6], keypoints[7]]
                    aligned_arm = [aligned_keypoints[5], aligned_keypoints[6], aligned_keypoints[7]]
                else:
                    orig_arm = []
                    aligned_arm = []
                
                def calc_arm_length(arm_kps):
                    if len(arm_kps) < 3:
                        return 0
                    dist1 = np.sqrt((arm_kps[1][0] - arm_kps[0][0])**2 + (arm_kps[1][1] - arm_kps[0][1])**2)
                    dist2 = np.sqrt((arm_kps[2][0] - arm_kps[1][0])**2 + (arm_kps[2][1] - arm_kps[1][1])**2)
                    return dist1 + dist2
                
                orig_length = calc_arm_length(orig_arm)
                aligned_length = calc_arm_length(aligned_arm)
                avg_length = (orig_length + aligned_length) / 2 if orig_length > 0 and aligned_length > 0 else 263
                
                scale = avg_length / 263
                thickness = max(40, int(80 * scale))
                radius = max(25, int(50 * scale))

                for i in range(len(orig_arm) - 1):
                    pt1, pt2 = tuple(map(int, orig_arm[i])), tuple(map(int, orig_arm[i + 1]))
                    cv2.line(mask_array, pt1, pt2, 255, thickness=thickness)
                for pt in orig_arm:
                    cv2.circle(mask_array, tuple(map(int, pt)), radius, 255, -1)
                
                for i in range(len(aligned_arm) - 1):
                    pt1, pt2 = tuple(map(int, aligned_arm[i])), tuple(map(int, aligned_arm[i + 1]))
                    cv2.line(mask_array, pt1, pt2, 255, thickness=thickness)
                for pt in aligned_arm:
                    cv2.circle(mask_array, tuple(map(int, pt)), radius, 255, -1)
                
                mask_array = cv2.GaussianBlur(mask_array, (71, 71), 0)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                mask_array = cv2.dilate(mask_array, kernel, iterations=1)
                
                mask_normalized = mask_array.astype(np.float32) / 255.0
                mask_binary = (mask_normalized > 0.5).astype(np.uint8)
                dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
                
                if dist_transform.max() > 0:
                    feather_width = 80
                    feather_mask = np.clip(dist_transform / feather_width, 0, 1)
                    mask_normalized = np.minimum(mask_normalized, feather_mask)
                
                mask_normalized = cv2.GaussianBlur(mask_normalized, (35, 35), 0)
                mask_final = (mask_normalized * 255).astype(np.uint8)
                
                mask_path_save = os.path.join(mask_dir, f'mask_{idx}.png')
                PILImage.fromarray(mask_final).save(mask_path_save)
                
                print(f"  ✓ Generated mask")
                
                processing_status[idx] = {'status': 'processing', 'step': 4, 'message': 'Step 4/4: AI 생성 중 (약 2-3분)...'}
                
                ipadapter_script = os.path.join(model_dir, 'generate', 'ipdemo.py')
                ipadapter_result = subprocess.run(
                    [sys.executable, ipadapter_script, '--idx', str(idx), '--steps', '30', '--guidance', '4.3', '--controlnet-scale', '1.2', '--ip-scale', '0.4', '--strength', '1.0', '--no-expand-arm', '--prompt', 'masterpiece, best quality, detailed hands, beautiful fingers, anatomically correct, matching reference clothing, consistent fabric texture and color'],
                    cwd=os.path.join(model_dir, 'generate'),
                    capture_output=True,
                    text=True,
                    timeout=400
                )
                
                if ipadapter_result.returncode != 0:
                    print(f"Generation error: {ipadapter_result.stderr}")
                    raise Exception(f"Generation failed: {ipadapter_result.stderr}")
                
                print(f"Generation output: {ipadapter_result.stdout}")
                
                generated_file = os.path.join(model_dir, '..', 'flask_app', 'data', 'results', f'{idx}_ipadapter.png')
                result_filename = f'result_{idx}.png'
                result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                if os.path.exists(generated_file):
                    import shutil
                    shutil.copy(generated_file, result_path)
                
                processing_status[idx] = {'status': 'completed', 'step': 4, 'message': '완료!', 'result': result_filename}
                
            except subprocess.TimeoutExpired:
                print("Processing timeout")
                processing_status[idx] = {'status': 'error', 'step': processing_status[idx]['step'], 'message': '시간 초과'}
                image_with_result = Image.new('RGBA', (1024, 1024))
                image_with_result.paste(image.convert('RGBA'), (0, 0))
                image_with_result.paste(sketch, (0, 0), sketch)
                result_filename = f"result_{base}.png"
                result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                image_with_result.save(result_path)
            except Exception as e:
                print(f"Processing failed: {e}")
                processing_status[idx] = {'status': 'error', 'step': processing_status[idx]['step'], 'message': f'오류: {str(e)}'}
                image_with_result = Image.new('RGBA', (1024, 1024))
                image_with_result.paste(image.convert('RGBA'), (0, 0))
                image_with_result.paste(sketch, (0, 0), sketch)
                result_filename = f"result_{base}.png"
                result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                image_with_result.save(result_path)
        
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()
        
        return render_template('index.html', image_path=orig_filename, processing=True, idx=base)
    abort(400)

@app.route('/status/<idx>')
def get_status(idx):
    """현재 처리 상태 확인 (polling용)"""
    if idx in processing_status:
        return jsonify(processing_status[idx])
    return jsonify({'status': 'not_found', 'message': '처리 정보를 찾을 수 없습니다.'})
if idx in processing_status:
        return jsonify(processing_status[idx])
    return jsonify({'status': 'not_found', 'message': '처리 정보 없음'})

@app.route('/stream/<idx>')
def stream_status(idx):
                current = processing_status[idx]
                if current != last_status:
                    yield f"data: {json.dumps(current)}\n\n"
                    last_status = current
                    if current['status'] in ['completed', 'error']:
                        break
            time.sleep(0.5)
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/sketch/<filename>')
def sketch_file(filename):
    return send_from_directory(app.config['SKETCH_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/pose/<filename>')
def pose_file(filename):
    """원본 포즈 이미지 제공"""
    pose_dir = os.path.join(model_dir, 'pose', 'pose_img')
    pose_dir = os.path.join(model_dir, 'pose', 'pose_img')
    return send_from_directory(pose_dir, filename)

@app.route('/aligned_pose/<filename>')
def aligned_pose_file(filename):
    pose_dir = os.path.join(model_dir, 'pose', 'pose_img')
    return send_from_directory(pose_dir, filename)

@app.route('/mask/<filename>')
def mask_file(filename):
    mask_dir = os.path.join(model_dir, 'mask', 'mask_img')
    return send_from_directory(mask_dir, filename)

if __name__ == '__main__':ebug=False, host='0.0.0.0', port=port, threaded=True)