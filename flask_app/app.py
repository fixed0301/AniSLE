from flask import Flask, request, render_template, send_from_directory, abort
from PIL import Image
from werkzeug.utils import secure_filename
import os
import glob

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
SKETCH_FOLDER = 'static/sketch'
RESULT_FOLDER = 'static/results'

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
                image = Image.open(file).convert('RGB').resize((256, 256), Image.Resampling.LANCZOS)
                image.save(file_path)
                return render_template('index.html', image_path=filename)
    return render_template('index.html', image_path=None)

@app.route('/save_sketch', methods=['POST'])
def save_sketch():
    # Blob 방식으로 넘어온 sketch 파일 처리
    if 'sketch' in request.files and 'filename' in request.form:
        file = request.files['sketch']
        sketch = Image.open(file.stream).convert('RGBA').resize((256, 256), Image.Resampling.LANCZOS)

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

        # TODO: AI 처리 로직
        # ai_result = my_ai_process(image, sketch)
        # image_with_ai = ai_result

        # 임시: 원본 + 스케치 합치기
        image_with_result = Image.new('RGBA', (256, 256))
        image_with_result.paste(image.convert('RGBA'), (0, 0))
        image_with_result.paste(sketch, (0, 0), sketch)

        result_filename = f"result_{base}.png"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        image_with_result.save(result_path)

        return render_template('index.html', image_path=orig_filename, result_path=result_filename)
    abort(400)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/sketch/<filename>')
def sketch_file(filename):
    return send_from_directory(app.config['SKETCH_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
