import { STORAGE_KEY } from './constants.js';
import { gridModeSelect, importJsonInput, northGuideSelect, uploadOverlay } from './dom.js';
import { state } from './state.js';
import { updateStatus } from './status.js';
import { updateQuadAnalysis } from './orientation.js';
import { resetView } from './image.js';
import { render } from './render.js';
import type { ProjectFile } from './types.js';

export function exportProjectJSON(): void {
  const projectData = {
    appName: "block-orientation-sampler",
    version: "1.1",
    exportedAt: new Date().toISOString(),
    gridMode: state.gridMode,
    northGuideAngle: state.northGuideAngle,
    imageDataUrl: state.imageDataUrl,
    quads: state.quads.map(q => ({
      id: q.id,
      points: q.points,
      placement: q.placement
    }))
  };

  const jsonStr = JSON.stringify(projectData, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `quadmatrix_project_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
  updateStatus('Exported');
}

// --- LocalStorage Persistence ---
export function saveStateToStorage(): void {
  try {
    const payload = {
      gridMode: state.gridMode,
      northGuideAngle: state.northGuideAngle,
      quads: state.quads,
      imageDataUrl: state.imageDataUrl
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch (err) {
    console.warn('LocalStorage save failed:', err);
  }
}

export function loadStateFromStorage(): void {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const data: ProjectFile = JSON.parse(raw);

    if (data.gridMode) {
      state.gridMode = data.gridMode;
      gridModeSelect.value = data.gridMode;
    }

    if (typeof data.northGuideAngle === 'number') {
      state.northGuideAngle = data.northGuideAngle;
      northGuideSelect.value = data.northGuideAngle.toString();
    }

    if (Array.isArray(data.quads)) {
      state.quads = data.quads;
    }

    if (data.imageDataUrl) {
      state.imageDataUrl = data.imageDataUrl;
      const img = new Image();
      img.onload = () => {
        state.image = img;
        state.imageLoaded = true;
        uploadOverlay.classList.add('hidden');
        resetView();

        state.quads.forEach(q => updateQuadAnalysis(q));
        updateStatus('Session restored');
      };
      img.src = data.imageDataUrl;
    }
  } catch (err) {
    console.warn('LocalStorage restore failed:', err);
  }
}

export function registerPersistenceListeners(): void {
  importJsonInput.addEventListener('change', (e) => {
    const input = e.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;
    const file = input.files[0];
    const reader = new FileReader();
    reader.onload = (evt) => {
      try {
        const result = evt.target?.result;
        if (typeof result !== 'string') throw new Error('unreadable file');
        const data: ProjectFile = JSON.parse(result);
        if (data.gridMode) {
          state.gridMode = data.gridMode;
          gridModeSelect.value = data.gridMode;
        }
        if (typeof data.northGuideAngle === 'number') {
          state.northGuideAngle = data.northGuideAngle;
          northGuideSelect.value = data.northGuideAngle.toString();
        }
        if (Array.isArray(data.quads)) {
          state.quads = data.quads;
        }
        if (data.imageDataUrl) {
          state.imageDataUrl = data.imageDataUrl;
          const img = new Image();
          img.onload = () => {
            state.image = img;
            state.imageLoaded = true;
            uploadOverlay.classList.add('hidden');
            resetView();
            state.quads.forEach(q => updateQuadAnalysis(q));
            saveStateToStorage();
            updateStatus('Imported');
          };
          img.src = data.imageDataUrl;
        } else {
          state.quads.forEach(q => updateQuadAnalysis(q));
          saveStateToStorage();
          requestAnimationFrame(render);
          updateStatus('Imported shapes');
        }
      } catch (err) {
        alert('Not a valid project file.');
        console.error(err);
      }
    };
    reader.readAsText(file);
    input.value = '';
  });
}
