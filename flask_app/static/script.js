let baseImage;
let sketchLayer;
let mainCanvas;

function setup() {
  mainCanvas = createCanvas(512, 512);
  mainCanvas.parent('canvasContainer');

  sketchLayer = createGraphics(512, 512);
  sketchLayer.clear();

  const imgTag = document.getElementById('baseImage');
  if (imgTag) {
    baseImage = loadImage(imgTag.src, () => {
      image(baseImage, 0, 0, width, height);
    });
  }
}

function draw() {
  if (!baseImage) return;
  image(baseImage, 0, 0, width, height);
  image(sketchLayer, 0, 0);

  // 마우스 드래그 시 스케치 레이어에만 선 그리기
  if (mouseIsPressed &&
      mouseX >= 0 && mouseX <= width &&
      mouseY >= 0 && mouseY <= height) {
    sketchLayer.stroke(255, 0, 0);
    sketchLayer.strokeWeight(5);
    sketchLayer.line(mouseX, mouseY, pmouseX, pmouseY);
  }
}

function saveSketch(event) {
  event.preventDefault();

  // 스케치 레이어만 PNG Blob으로 변환
  sketchLayer.elt.toBlob(blob => {
    const formData = new FormData();
    formData.append('filename', filename);
    formData.append('sketch', blob, 'sketch.png');

    fetch('/save_sketch', { method: 'POST', body: formData })
      .then(res => res.text())
      .then(html => {
        document.open();
        document.write(html);
        document.close();
      })
      .catch(console.error);
  }, 'image/png');
}

window.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('sketchForm');
  if (form) form.addEventListener('submit', saveSketch);
});
