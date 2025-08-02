let sketchCanvas;
let uploadedImage;

const sketch = (p) => {
  p.preload = () => {
    // preload에서 이미지 로딩은 이벤트 리스너에서 처리하므로 생략
  };

  p.setup = () => {
    sketchCanvas = p.createCanvas(500, 500);
    sketchCanvas.parent('canvasContainer');
    p.background(0, 0, 0, 0); // 투명 배경
    if (uploadedImage) {
      p.image(uploadedImage, 0, 0, 500, 500); // 초기 이미지 표시
    }
  };

  p.draw = () => {
    if (uploadedImage) {
      p.image(uploadedImage, 0, 0, 500, 500); // 배경으로 이미지 항상 표시
    }
    if (p.mouseIsPressed) {
      p.stroke(255, 0, 0); // 빨간색 펜
      p.strokeWeight(3);
      p.line(p.mouseX, p.mouseY, p.pmouseX, p.pmouseY); // 이미지 위에 선 그리기
    }
  };
};

// p5.js 인스턴스 모드 초기화
new p5(sketch);

document.getElementById('imageInput').addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      uploadedImage = p5.loadImage(e.target.result, () => {
        console.log('Image loaded successfully');
        // 이미지 로드 후 캔버스 갱신
        const p = new p5(sketch); // 새 p5 인스턴스 생성 및 갱신
        if (uploadedImage.width > 0 && uploadedImage.height > 0) {
          let aspectRatio = uploadedImage.width / uploadedImage.height;
          let newWidth = 500;
          let newHeight = 500 / aspectRatio;
          if (newHeight > 500) {
            newHeight = 500;
            newWidth = 500 * aspectRatio;
          }
          sketchCanvas = p.createCanvas(newWidth, newHeight);
          sketchCanvas.parent('canvasContainer');
          p.background(0, 0, 0, 0);
          p.image(uploadedImage, 0, 0, newWidth, newHeight);
        }
      }, () => {
        console.error('Image loading failed');
      });
    };
    reader.readAsDataURL(file);
  }
});

function saveImages() {
  if (uploadedImage && sketchCanvas) {
    // 원본 이미지 다운로드
    const originalLink = document.createElement('a');
    originalLink.href = uploadedImage.canvas.toDataURL('image/png');
    originalLink.download = 'original.png';
    originalLink.click();

    // 스케치 이미지 다운로드
    const sketchLink = document.createElement('a');
    sketchLink.href = sketchCanvas.elt.toDataURL('image/png');
    sketchLink.download = 'sketch.png';
    sketchLink.click();
  } else {
    alert('이미지를 먼저 업로드하세요.');
  }
}

async function processWithAI() {
  if (uploadedImage && sketchCanvas) {
    const formData = new FormData();
    formData.append('original', dataURLtoBlob(uploadedImage.canvas.toDataURL('image/png')));
    formData.append('sketch', dataURLtoBlob(sketchCanvas.elt.toDataURL('image/png')));

    try {
      const response = await fetch('/upload', {
        method: 'POST',
        body: formData
      });
      const result = await response.json();

      if (result.success) {
        const resultContainer = document.getElementById('resultContainer');
        const resultImage = document.getElementById('resultImage');
        resultImage.src = result.ai_result;
        resultContainer.style.display = 'block';

        // 서버에서 다운로드
        ['original', 'sketch', 'ai_result'].forEach(file => {
          const link = document.createElement('a');
          link.href = result[file];
          link.download = file.split('/').pop();
          link.click();
        });
      } else {
        alert('Error: ' + result.error);
      }
    } catch (error) {
      alert('서버 오류: ' + error.message);
    }
  } else {
    alert('이미지를 먼저 업로드하세요.');
  }
}

function dataURLtoBlob(dataURL) {
  const [header, data] = dataURL.split(',');
  const mime = header.match(/:(.*?);/)[1];
  const binary = atob(data);
  const array = [];
  for (let i = 0; i < binary.length; i++) {
    array.push(binary.charCodeAt(i));
  }
  return new Blob([new Uint8Array(array)], { type: mime });
}