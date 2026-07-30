import { MIN_SCALE } from './constants.js';
import { canvas, fileInput, uploadBox, uploadOverlay, viewport } from './dom.js';
import { state } from './state.js';
import { updateStatus } from './status.js';
import { saveStateToStorage } from './persistence.js';
import { resetHistory } from './history.js';
import { render } from './render.js';

// --- Canvas Setup ---
export function resizeCanvas(): void {
  const rect = viewport.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  requestAnimationFrame(render);
}

// --- Image Loading ---
export function loadImageFromFile(file: File | null | undefined): void {
  if (!file || !file.type.startsWith('image/')) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    const dataUrl = e.target?.result;
    if (typeof dataUrl !== 'string') return;
    const img = new Image();
    img.onload = () => {
      state.image = img;
      state.imageDataUrl = dataUrl;
      state.imageLoaded = true;
      state.quads = [];
      state.currentPoints = [];
      state.copyingQuad = null;
      uploadOverlay.classList.add('hidden');
      resetView();
      saveStateToStorage();
      // A new image is a new document — there is nothing behind it to walk back into.
      resetHistory();
      updateStatus('Image loaded');
    };
    img.src = dataUrl;
  };
  reader.readAsDataURL(file);
}

export function resetView(): void {
  if (!state.image) return;
  const vw = canvas.width;
  const vh = canvas.height;
  const imgW = state.image.width;
  const imgH = state.image.height;

  // The margin is what keeps the image off the edges of the viewport, but a fixed 60px
  // is more than a small viewport has to give: below 120px of width or height it
  // consumes the whole axis and the fitted scale comes out zero or negative, which
  // mirrors the image and makes the view unusable. Cap it at a fifth of each axis so it
  // always leaves room, and clamp the result to the same floor the wheel handler uses.
  const margin = Math.min(60, vw / 5, vh / 5);
  const scaleW = (vw - margin * 2) / imgW;
  const scaleH = (vh - margin * 2) / imgH;
  state.scale = Math.max(MIN_SCALE, Math.min(scaleW, scaleH, 1.0));

  state.panX = (vw - imgW * state.scale) / 2;
  state.panY = (vh - imgH * state.scale) / 2;
  requestAnimationFrame(render);
}

export function registerImageListeners(): void {
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  fileInput.addEventListener('change', (e) => {
    const files = (e.target as HTMLInputElement).files;
    if (files && files.length > 0) {
      loadImageFromFile(files[0]);
    }
  });

  uploadBox.addEventListener('click', () => fileInput.click());
  viewport.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
  });
  viewport.addEventListener('dragleave', () => uploadBox.classList.remove('dragover'));
  viewport.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      loadImageFromFile(files[0]);
    }
  });
}
