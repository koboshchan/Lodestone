import {
  globalContextMenu, menuConfidenceBadge, menuCopy, menuDownload, menuOrientationBadge,
  menuPlacement, menuPreviewCanvas, menuRemove, menuScoresGrid, menuUndoPlacement,
  quadContextMenu
} from './dom.js';
import { state } from './state.js';
import { updateStatus } from './status.js';
import { updateQuadAnalysis } from './orientation.js';
import { applyPlacement, solvePlacement, undoPlacement } from './placement/solve.js';
import { saveStateToStorage } from './persistence.js';
import { drawNorthArrowOverlay, render } from './render.js';
import type { Quad } from './types.js';

export function positionAndShowMenu(
  menuElem: HTMLElement, clientX: number, clientY: number
): void {
  closeAllContextMenus();

  // Reset top/left/maxHeight so layout engine calculates unconstrained intrinsic size
  menuElem.style.top = '0px';
  menuElem.style.left = '0px';
  menuElem.style.maxHeight = 'none';
  menuElem.style.visibility = 'hidden';
  menuElem.style.display = 'flex';
  menuElem.classList.add('active');

  // Query true unclipped intrinsic width & height
  const menuWidth = menuElem.offsetWidth || 280;
  const menuHeight = menuElem.offsetHeight || 320;

  // Restore CSS max-height constraint
  menuElem.style.maxHeight = 'calc(100vh - 20px)';

  // Calculate safe target X coordinate
  let posX = clientX;
  if (posX + menuWidth > window.innerWidth - 10) {
    posX = window.innerWidth - menuWidth - 10;
  }
  posX = Math.max(10, posX);

  // Calculate safe target Y coordinate
  let posY = clientY;
  if (posY + menuHeight > window.innerHeight - 10) {
    posY = window.innerHeight - menuHeight - 10;
  }
  posY = Math.max(10, posY);

  menuElem.style.left = `${posX}px`;
  menuElem.style.top = `${posY}px`;
  menuElem.style.visibility = 'visible';
}

export function openQuadContextMenu(x: number, y: number, quad: Quad | undefined): void {
  if (!quad) return;

  if (!quad.analysisResult) {
    updateQuadAnalysis(quad);
  }

  const res = quad.analysisResult;
  if (!res) return;
  const analysis = res.analysis;

  const pCtx = menuPreviewCanvas.getContext('2d');
  if (!pCtx) return;
  pCtx.imageSmoothingEnabled = false;
  pCtx.clearRect(0, 0, 80, 80);
  pCtx.drawImage(res.canvas, 0, 0, 80, 80);

  drawNorthArrowOverlay(menuPreviewCanvas, analysis.angle);

  menuOrientationBadge.textContent = analysis.bestOrientation;
  menuConfidenceBadge.textContent = `conf ${analysis.confidence}% · facing ${analysis.facing}`;

  menuUndoPlacement.style.display = quad.placement ? 'flex' : 'none';
  if (quad.placement) {
    const lat = quad.placement.lattice;
    menuConfidenceBadge.textContent +=
      ` · ${lat.nu}×${lat.nv} fit ${Math.round(lat.score * 100)}%`;
  }

  menuScoresGrid.innerHTML = analysis.scores.map(s => `
        <div class="score-item ${s.name === analysis.bestOrientation ? 'active' : ''}">
          <span>${s.code}:</span>
          <span>${(s.corr * 100).toFixed(1)}%</span>
        </div>
      `).join('');

  positionAndShowMenu(quadContextMenu, x, y);

  // Preserve contextQuadId after menu opens!
  state.contextQuadId = quad.id;
}

export function openGlobalContextMenu(x: number, y: number): void {
  positionAndShowMenu(globalContextMenu, x, y);
  state.contextQuadId = null;
}

export function closeAllContextMenus(): void {
  quadContextMenu.classList.remove('active');
  quadContextMenu.style.display = 'none';
  globalContextMenu.classList.remove('active');
  globalContextMenu.style.display = 'none';
}

export function registerMenuListeners(): void {
  menuPlacement.addEventListener('click', () => {
    if (!state.contextQuadId) return;
    const quad = state.quads.find(q => q.id === state.contextQuadId);
    closeAllContextMenus();
    if (!quad) return;

    // The search runs for a few hundred milliseconds and blocks the page while it
    // does, so hand the status line a frame to paint before starting.
    updateStatus('Solving…');
    setTimeout(() => {
      const solution = solvePlacement(quad);
      if (!solution) {
        updateStatus('No lattice found');
        return;
      }

      const created = applyPlacement(quad, solution);
      const lat = solution.lattice;
      updateStatus(`${lat.nu}×${lat.nv} blocks · fit ${Math.round(lat.score * 100)}%`);
      state.contextQuadId = created.length ? created[0].id : null;
      saveStateToStorage();
      requestAnimationFrame(render);
    }, 0);
  });

  menuUndoPlacement.addEventListener('click', () => {
    if (!state.contextQuadId) return;
    const quad = state.quads.find(q => q.id === state.contextQuadId);
    closeAllContextMenus();
    if (!quad || !quad.placement) return;

    undoPlacement(quad.placement.groupId);
    state.contextQuadId = null;
    updateStatus('Placement undone');
    saveStateToStorage();
    requestAnimationFrame(render);
  });

  menuDownload.addEventListener('click', () => {
    if (!state.contextQuadId) return;
    const quad = state.quads.find(q => q.id === state.contextQuadId);
    if (quad && quad.analysisResult) {
      const link = document.createElement('a');
      link.download = `grass_block_16x16_${quad.id}.png`;
      link.href = quad.analysisResult.dataUrl;
      link.click();
    }
    closeAllContextMenus();
  });

  menuCopy.addEventListener('click', () => {
    if (!state.contextQuadId) return;
    const quad = state.quads.find(q => q.id === state.contextQuadId);
    if (quad) {
      const cx = (quad.points[0].x + quad.points[1].x + quad.points[2].x + quad.points[3].x) / 4;
      const cy = (quad.points[0].y + quad.points[1].y + quad.points[2].y + quad.points[3].y) / 4;

      state.copyingQuad = {
        points: JSON.parse(JSON.stringify(quad.points)),
        anchor: { x: cx, y: cy }
      };
      updateStatus('Click to place duplicate');
    }
    closeAllContextMenus();
    requestAnimationFrame(render);
  });

  menuRemove.addEventListener('click', () => {
    if (!state.contextQuadId) return;
    state.quads = state.quads.filter(q => q.id !== state.contextQuadId);
    updateStatus('Removed');
    closeAllContextMenus();
    saveStateToStorage();
    requestAnimationFrame(render);
  });
}
