import { IS_MAC, MAX_SCALE, MIN_SCALE } from './constants.js';
import {
  canvas, globalClearShapes, globalExportJson, globalImportJsonLabel, globalLoadImgLabel,
  globalResetView, gridModeSelect, northGuideSelect, viewport
} from './dom.js';
import { state } from './state.js';
import { updateStatus } from './status.js';
import {
  findCornerHandleNearScreen, imageToScreen, isPassThrough, isPointInQuad,
  screenToImage, sortQuadPoints
} from './geometry.js';
import { updateQuadAnalysis } from './orientation.js';
import { exportProjectJSON, saveStateToStorage } from './persistence.js';
import { resetView } from './image.js';
import { closeAllContextMenus, openGlobalContextMenu, openQuadContextMenu } from './menu.js';
import { render } from './render.js';
import type { GridMode, Quad } from './types.js';

// Taking or releasing the modifier changes what a click will do, so the hover
// affordance has to update even if the pointer never moves.
function refreshHoverState(passThrough: boolean): void {
  if (state.draggedHandle || state.isPanning || !state.imageLoaded) return;

  const scr = imageToScreen(state.mouseImgPos.x, state.mouseImgPos.y);
  const hit = passThrough ? null : findCornerHandleNearScreen(scr.x, scr.y);
  const changed = (hit && hit.quadId) !== (state.hoveredHandle && state.hoveredHandle.quadId) ||
    (hit && hit.pointIdx) !== (state.hoveredHandle && state.hoveredHandle.pointIdx);

  state.hoveredHandle = hit;
  viewport.style.cursor = hit ? 'pointer' : 'crosshair';
  if (changed) requestAnimationFrame(render);
}

function registerControlListeners(): void {
  gridModeSelect.addEventListener('change', (e) => {
    const value = (e.target as HTMLSelectElement).value as GridMode;
    state.gridMode = value;
    updateStatus(`Projection: ${value === 'homography' ? 'perspective' : 'bilinear'}`);
    state.quads.forEach(q => updateQuadAnalysis(q));
    saveStateToStorage();
    requestAnimationFrame(render);
  });

  northGuideSelect.addEventListener('change', (e) => {
    const select = e.target as HTMLSelectElement;
    state.northGuideAngle = parseInt(select.value, 10);
    updateStatus(`North: ${select.options[select.selectedIndex].text.toLowerCase()} edge`);
    state.quads.forEach(q => updateQuadAnalysis(q));
    saveStateToStorage();
    requestAnimationFrame(render);
  });

  globalLoadImgLabel.addEventListener('click', () => closeAllContextMenus());
  globalImportJsonLabel.addEventListener('click', () => closeAllContextMenus());

  globalExportJson.addEventListener('click', () => {
    exportProjectJSON();
    closeAllContextMenus();
  });

  globalResetView.addEventListener('click', () => {
    resetView();
    closeAllContextMenus();
  });

  globalClearShapes.addEventListener('click', () => {
    state.quads = [];
    state.currentPoints = [];
    state.copyingQuad = null;
    state.draggedHandle = null;
    saveStateToStorage();
    requestAnimationFrame(render);
    updateStatus('Cleared');
    closeAllContextMenus();
  });
}

// --- Interaction Listeners ---
export function registerInteractionListeners(): void {
  registerControlListeners();

  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !state.isSpacePressed) {
      state.isSpacePressed = true;
      viewport.classList.add('panning');
    }
    if (e.key === (IS_MAC ? 'Meta' : 'Control')) refreshHoverState(true);
  });

  window.addEventListener('keyup', (e) => {
    if (e.code === 'Space') {
      state.isSpacePressed = false;
      state.isPanning = false;
      viewport.classList.remove('panning');
    }
    if (e.key === (IS_MAC ? 'Meta' : 'Control')) refreshHoverState(false);
  });

  viewport.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    state.mouseImgPos = screenToImage(mouseX, mouseY);

    const dragged = state.draggedHandle;
    if (dragged) {
      const quad = state.quads.find(q => q.id === dragged.quadId);
      if (quad) {
        quad.points[dragged.pointIdx] = { ...state.mouseImgPos };
        requestAnimationFrame(render);
      }
      return;
    }

    if (state.isPanning) {
      state.panX = mouseX - state.panStartX;
      state.panY = mouseY - state.panStartY;
      requestAnimationFrame(render);
      return;
    }

    const wasHovering = !!state.hoveredHandle;
    const handleHit = isPassThrough(e) ? null : findCornerHandleNearScreen(mouseX, mouseY);
    if (handleHit) {
      state.hoveredHandle = handleHit;
      viewport.style.cursor = 'pointer';
    } else {
      if (state.hoveredHandle) {
        state.hoveredHandle = null;
        viewport.style.cursor = 'crosshair';
      }
    }

    // The ticked rubber-band edge follows the cursor from the first point on.
    if (wasHovering !== !!state.hoveredHandle ||
      state.currentPoints.length > 0 || state.copyingQuad) {
      requestAnimationFrame(render);
    }
  });

  viewport.addEventListener('wheel', (e) => {
    e.preventDefault();
    if (!state.imageLoaded) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const zoomFactor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    const newScale = Math.min(Math.max(state.scale * zoomFactor, MIN_SCALE), MAX_SCALE);

    state.panX = mouseX - (mouseX - state.panX) * (newScale / state.scale);
    state.panY = mouseY - (mouseY - state.panY) * (newScale / state.scale);
    state.scale = newScale;

    requestAnimationFrame(render);
  }, { passive: false });

  viewport.addEventListener('mousedown', (e) => {
    closeAllContextMenus();

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    if (state.isSpacePressed || e.button === 1) {
      state.isPanning = true;
      state.panStartX = mouseX - state.panX;
      state.panStartY = mouseY - state.panY;
      return;
    }

    if (e.button === 0 && state.imageLoaded) {
      // Holding cmd/ctrl passes through existing corner handles so a point can be
      // placed on top of one instead of grabbing it.
      const handleHit = isPassThrough(e) ? null : findCornerHandleNearScreen(mouseX, mouseY);
      if (handleHit) {
        state.draggedHandle = handleHit;
        viewport.classList.add('dragging-handle');
        updateStatus('Dragging corner');
        return;
      }

      const imgPt = screenToImage(mouseX, mouseY);

      const copying = state.copyingQuad;
      if (copying) {
        const placedPoints = copying.points.map(p => ({
          x: p.x + (imgPt.x - copying.anchor.x),
          y: p.y + (imgPt.y - copying.anchor.y)
        }));

        const newQuad: Quad = {
          id: Date.now(),
          points: sortQuadPoints(placedPoints)
        };

        updateQuadAnalysis(newQuad);
        state.quads.push(newQuad);

        state.copyingQuad = null;
        updateStatus('Duplicate placed');
        saveStateToStorage();
        requestAnimationFrame(render);
        return;
      }

      if (state.currentPoints.length < 3) {
        state.currentPoints.push(imgPt);
        updateStatus(`Point ${state.currentPoints.length}/4`);
      } else if (state.currentPoints.length === 3) {
        state.currentPoints.push(imgPt);
        const sortedQuad = sortQuadPoints(state.currentPoints);

        const newQuad: Quad = {
          id: Date.now(),
          points: sortedQuad
        };

        updateQuadAnalysis(newQuad);
        state.quads.push(newQuad);

        state.currentPoints = [];
        updateStatus(`Quad added · ${newQuad.analysisResult?.analysis.bestOrientation}`);
        saveStateToStorage();
      }

      requestAnimationFrame(render);
    }
  });

  window.addEventListener('mouseup', () => {
    const dragged = state.draggedHandle;
    if (dragged) {
      const quad = state.quads.find(q => q.id === dragged.quadId);
      if (quad) {
        updateQuadAnalysis(quad);
      }
      state.draggedHandle = null;
      viewport.classList.remove('dragging-handle');
      updateStatus('Re-analyzed');
      saveStateToStorage();
      requestAnimationFrame(render);
    }
    if (state.isPanning && !state.isSpacePressed) {
      state.isPanning = false;
    }
  });

  // Right Click Event Handling: Quad Menu vs Empty Space Global Menu
  viewport.addEventListener('contextmenu', (e) => {
    e.preventDefault();

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const imgPt = screenToImage(mouseX, mouseY);

    let hitQuad: Quad | null = null;
    if (state.imageLoaded) {
      for (let i = state.quads.length - 1; i >= 0; i--) {
        if (isPointInQuad(imgPt, state.quads[i].points)) {
          hitQuad = state.quads[i];
          break;
        }
      }
    }

    if (hitQuad) {
      openQuadContextMenu(e.clientX, e.clientY, hitQuad);
    } else {
      openGlobalContextMenu(e.clientX, e.clientY);
    }
  });
}
