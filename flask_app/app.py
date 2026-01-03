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

# Add model directories to path
model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
sys.path.insert(0, model_dir)

app = Flask(__name__)

# Global dictionary to track processing status
processing_status = {}

# flask_app 디렉토리 내의 data 폴더 사용
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
    """업로드 폴더에서 다음 인덱스 번호 반환"""
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
    """서버 상태 확인용"""
    return {'status': 'ok', 'message': 'Flask server is running'}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename:
                # 다음 인덱스 번호 가져오기
                index = get_next_index()
                ext = os.path.splitext(file.filename)[1]
                filename = f"{index}{ext}"
                
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image = Image.open(file).convert('RGB').resize((512, 512), Image.Resampling.LANCZOS)
                image.save(file_path)
                
                # 포즈 추출
                try:
                    print(f"Extracting pose for {filename}...")
                    import sys
                    sys.path.insert(0, model_dir)
                    from openpose_simple import extract_pose
                    
                    pose_img, keypoints = extract_pose(image)
                    
                    # 포즈 이미지 저장
                    pose_img_path = os.path.join(model_dir, 'pose', 'pose_img', f'{index}.png')
                    os.makedirs(os.path.dirname(pose_img_path), exist_ok=True)
                    pose_img.save(pose_img_path)
                    
                    # Keypoints JSON 저장
                    keypoints_path = os.path.join(model_dir, 'pose', 'pose_json', f'keypoints_{index}.json')
                    os.makedirs(os.path.dirname(keypoints_path), exist_ok=True)
                    import json
                    with open(keypoints_path, 'w') as f:
                        json.dump(keypoints, f, indent=2)
                    
                    print(f"✓ Pose extracted and saved to {pose_img_path}")
                    print(f"✓ Keypoints saved to {keypoints_path}")
                except Exception as e:
                    print(f"Warning: Pose extraction failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # 포즈 추출 실패해도 계속 진행
                
                # POST 후 GET으로 리다이렉트 (브라우저 새로고침 시 재전송 방지)
                return redirect(url_for('show_image', filename=filename))
    return render_template('index.html', image_path=None)

@app.route('/image/<filename>')
def show_image(filename):
    """업로드된 이미지를 표시하는 페이지"""
    # 결과 파일이 있는지 확인
    base = os.path.splitext(filename)[0]
    result_filename = f"result_{base}.png"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    
    if os.path.exists(result_path):
        return render_template('index.html', image_path=filename, result_path=result_filename)
    else:
        return render_template('index.html', image_path=filename)

@app.route('/save_sketch', methods=['POST'])
def save_sketch():
    # Blob 방식으로 넘어온 sketch 파일 처리
    if 'sketch' in request.files and 'filename' in request.form:
        file = request.files['sketch']
        sketch = Image.open(file.stream).convert('RGBA').resize((512, 512), Image.Resampling.LANCZOS)

        # 스케치만 별도 저장
        orig_filename = secure_filename(request.form['filename'])
        base, _ = os.path.splitext(orig_filename)
        sketch_filename = f"sketch_{base}.png"
        sketch_path = os.path.join(app.config['SKETCH_FOLDER'], sketch_filename)
        sketch.save(sketch_path)

        # 원본 이미지 불러오기
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], orig_filename)
        if not os.path.exists(image_path):
            abort(404, description="Original image not found.")
        image = Image.open(image_path).convert('RGB')

        # AI 처리를 백그라운드에서 실행
        idx = base
        processing_status[idx] = {'status': 'queued', 'step': 0, 'message': '처리 대기 중...'}
        
        def process_in_background():
            try:
                # 통합 파이프라인 실행 (포즈 추출 → 정렬 → 마스크 → 생성)
                print(f"Running full pipeline for idx={idx}")
                
                # Step 1: 포즈 로드
                processing_status[idx] = {'status': 'processing', 'step': 1, 'message': 'Step 1/4: 포즈 keypoints 로드 중...'}
                import time
                time.sleep(0.5)
                
                # Step 2: 정렬
                processing_status[idx] = {'status': 'processing', 'step': 2, 'message': 'Step 2/4: 스케치에 포즈 정렬 중...'}
                time.sleep(0.5)
                
                # Step 3: 마스크 생성
                processing_status[idx] = {'status': 'processing', 'step': 3, 'message': 'Step 3/4: 마스크 생성 중...'}
                
                pipeline_script = os.path.join(model_dir, 'pipeline.py')
                pipeline_result = subprocess.run(
                    [sys.executable, pipeline_script, '--idx', str(idx)],
                    cwd=model_dir,
                    capture_output=True,
                    text=True,
                    timeout=400  # 6-7 minutes total
                )
                
                # Step 4: 이미지 생성
                if pipeline_result.returncode == 0:
                    processing_status[idx] = {'status': 'processing', 'step': 4, 'message': 'Step 4/4: AI 이미지 생성 중 (약 2-3분 소요)...'}
                
                if pipeline_result.returncode != 0:
                    print(f"Pipeline error: {pipeline_result.stderr}")
                    raise Exception(f"Pipeline failed: {pipeline_result.stderr}")
                
                print(f"Pipeline output: {pipeline_result.stdout}")
                processing_status[idx] = {'status': 'completed', 'step': 3, 'message': '처리 완료!', 'result': f'result_{idx}.png'}
                
            except subprocess.TimeoutExpired:
                print("AI processing timeout")
                processing_status[idx] = {'status': 'error', 'step': processing_status[idx]['step'], 'message': '처리 시간 초과'}
                # Fallback: 원본 + 스케치 합성
                image_with_result = Image.new('RGBA', (512, 512))
                image_with_result.paste(image.convert('RGBA'), (0, 0))
                image_with_result.paste(sketch, (0, 0), sketch)
                result_filename = f"result_{base}.png"
                result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                image_with_result.save(result_path)
            except Exception as e:
                print(f"AI processing failed: {e}")
                processing_status[idx] = {'status': 'error', 'step': processing_status[idx]['step'], 'message': f'오류 발생: {str(e)}'}
                # Fallback: 원본 + 스케치 합성
                image_with_result = Image.new('RGBA', (512, 512))
                image_with_result.paste(image.convert('RGBA'), (0, 0))
                image_with_result.paste(sketch, (0, 0), sketch)
                result_filename = f"result_{base}.png"
                result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                image_with_result.save(result_path)
        
        # 백그라운드 스레드 시작
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()
        
        # 처리 시작 페이지 반환 (진행상황 표시)
        return render_template('index.html', image_path=orig_filename, processing=True, idx=base)
    abort(400)

@app.route('/status/<idx>')
def get_status(idx):
    """현재 처리 상태 확인 (polling용)"""
    if idx in processing_status:
        return jsonify(processing_status[idx])
    return jsonify({'status': 'not_found', 'message': '처리 정보를 찾을 수 없습니다.'})

@app.route('/stream/<idx>')
def stream_status(idx):
    """Server-Sent Events로 실시간 진행상황 전송"""
    def generate():
        last_status = None
        while True:
            if idx in processing_status:
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
    return send_from_directory(pose_dir, filename)

@app.route('/aligned_pose/<filename>')
def aligned_pose_file(filename):
    """정렬된 포즈 이미지 제공"""
    pose_dir = os.path.join(model_dir, 'pose', 'pose_img')
    return send_from_directory(pose_dir, filename)

@app.route('/mask/<filename>')
def mask_file(filename):
    """마스크 이미지 제공"""
    mask_dir = os.path.join(model_dir, 'mask', 'mask_img')
    return send_from_directory(mask_dir, filename)

if __name__ == '__main__':
    # RunPod HTTP Service용 - 포트 변경 가능
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
