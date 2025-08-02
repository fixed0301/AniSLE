from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import os
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 2048 * 2048
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename:
                filename = file.filename  # 원본 파일명 사용
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                # 이미지를 256x256으로 리사이즈하여 저장
                image = Image.open(file).convert('RGB').resize((256, 256), Image.Resampling.LANCZOS)
                image.save(file_path)
                return render_template('index.html', image_path=filename)
    return render_template('index.html', image_path=None)


@app.route('/save_sketch', methods=['POST'])
def save_sketch():
    if 'sketch' in request.form:
        sketch_data = request.form['sketch'].replace('data:image/png;base64,', '')
        sketch_data = sketch_data.replace(' ', '+')
        sketch = Image.open(io.BytesIO(base64.b64decode(sketch_data))).resize((256, 256), Image.Resampling.LANCZOS)
        filename = request.form['filename']
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image = Image.open(image_path).convert('RGB')

        # 스케치와 이미지를 합침
        image_with_sketch = Image.new('RGB', (256, 256))
        image_with_sketch.paste(image, (0, 0))
        sketch = sketch.convert('RGBA')
        image_with_sketch.paste(sketch, (0, 0), sketch)

        result_filename = f"result_{filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        image_with_sketch.save(result_path)

        return render_template('index.html', image_path=filename, result_path=result_filename)
    return render_template('index.html', image_path=None)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)