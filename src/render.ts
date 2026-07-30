import { ACCENT } from './constants.js';
import { canvas, ctx } from './dom.js';
import { state } from './state.js';
import {
  computeHomography, evaluateQuadPoint, sortQuadPoints, transformHomography
} from './geometry.js';
import type { DrawableQuad, Placement, Point } from './types.js';

// --- Overlay North Arrow helper for thumbnails ---
export function drawNorthArrowOverlay(
  previewCanvasElem: HTMLCanvasElement, angleDegrees: number
): void {
  const pCtx = previewCanvasElem.getContext('2d');
  if (!pCtx) return;
  const w = previewCanvasElem.width;
  const h = previewCanvasElem.height;
  const cx = w / 2;
  const cy = h / 2;

  pCtx.save();
  pCtx.translate(cx, cy);
  pCtx.rotate((angleDegrees * Math.PI) / 180);

  const arrowLen = w * 0.36;
  const headSize = w * 0.16;

  pCtx.strokeStyle = ACCENT;
  pCtx.lineWidth = Math.max(3, w * 0.07);
  pCtx.beginPath();
  pCtx.moveTo(0, arrowLen * 0.45);
  pCtx.lineTo(0, -arrowLen * 0.4);
  pCtx.stroke();

  pCtx.fillStyle = ACCENT;
  pCtx.beginPath();
  pCtx.moveTo(0, -arrowLen * 0.52);
  pCtx.lineTo(-headSize, -arrowLen * 0.12);
  pCtx.lineTo(headSize, -arrowLen * 0.12);
  pCtx.closePath();
  pCtx.fill();

  pCtx.fillStyle = '#ffffff';
  pCtx.font = `900 ${Math.round(w * 0.2)}px sans-serif`;
  pCtx.textAlign = 'center';
  pCtx.textBaseline = 'middle';
  pCtx.fillText('N', 0, -arrowLen * 0.72);

  pCtx.restore();
}

// --- Draw North Arrow ACCORDING TO QUADRILATERAL PERSPECTIVE SURFACE ---
export function drawNorthArrowOnQuadCanvas(
  targetCtx: CanvasRenderingContext2D, quadPts: Point[], angleDegrees: number
): void {
  if (!quadPts || quadPts.length !== 4) return;

  const effectiveAngle = (angleDegrees + state.northGuideAngle) % 360;

  const uCenter = 0.5, vCenter = 0.5;
  let uTip = 0.5, vTip = 0.18;
  let uLeft = 0.42, vLeft = 0.30;
  let uRight = 0.58, vRight = 0.30;

  if (effectiveAngle === 90) { // Right Edge
    uTip = 0.82; vTip = 0.5;
    uLeft = 0.70; vLeft = 0.42;
    uRight = 0.70; vRight = 0.58;
  } else if (effectiveAngle === 180) { // Bottom Edge
    uTip = 0.5; vTip = 0.82;
    uLeft = 0.58; vLeft = 0.70;
    uRight = 0.42; vRight = 0.70;
  } else if (effectiveAngle === 270) { // Left Edge
    uTip = 0.18; vTip = 0.5;
    uLeft = 0.30; vLeft = 0.58;
    uRight = 0.30; vRight = 0.42;
  }

  const pCenter = evaluateQuadPoint(quadPts, uCenter, vCenter);
  const pTip = evaluateQuadPoint(quadPts, uTip, vTip);
  const pLeft = evaluateQuadPoint(quadPts, uLeft, vLeft);
  const pRight = evaluateQuadPoint(quadPts, uRight, vRight);

  targetCtx.save();

  targetCtx.shadowColor = 'rgba(0, 0, 0, 0.9)';
  targetCtx.shadowBlur = 6 / state.scale;

  // Red Arrow Stem
  targetCtx.strokeStyle = ACCENT;
  targetCtx.lineWidth = 3.5 / state.scale;
  targetCtx.beginPath();
  targetCtx.moveTo(pCenter.x, pCenter.y);
  targetCtx.lineTo(pTip.x, pTip.y);
  targetCtx.stroke();

  // Red Arrow Head
  targetCtx.fillStyle = ACCENT;
  targetCtx.beginPath();
  targetCtx.moveTo(pTip.x, pTip.y);
  targetCtx.lineTo(pLeft.x, pLeft.y);
  targetCtx.lineTo(pRight.x, pRight.y);
  targetCtx.closePath();
  targetCtx.fill();

  // White 'N' Label
  const labelVecX = pTip.x - pCenter.x;
  const labelVecY = pTip.y - pCenter.y;
  const textX = pTip.x + labelVecX * 0.25;
  const textY = pTip.y + labelVecY * 0.25;

  targetCtx.fillStyle = '#ffffff';
  targetCtx.font = `900 ${Math.max(12 / state.scale, 14)}px sans-serif`;
  targetCtx.textAlign = 'center';
  targetCtx.textBaseline = 'middle';
  targetCtx.fillText('N', textX, textY);

  targetCtx.restore();
}

// --- Main Rendering Loop ---
export function render(): void {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!state.imageLoaded || !state.image) return;

  ctx.save();
  ctx.translate(state.panX, state.panY);
  ctx.scale(state.scale, state.scale);

  // 1. Draw Image
  ctx.drawImage(state.image, 0, 0);

  // 2. Draw Quadrilaterals
  drawPlacementLattices();
  state.quads.forEach((quad) => {
    drawQuadrilateralAndGrid(quad, ACCENT, 0.9);
  });

  // 3. Draw Draft Creation State
  if (state.currentPoints.length > 0) {
    ctx.strokeStyle = '#ffffff';
    ctx.fillStyle = '#ffffff';
    ctx.lineWidth = 2 / state.scale;

    state.currentPoints.forEach((pt, idx) => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 5 / state.scale, 0, Math.PI * 2);
      ctx.fill();

      ctx.font = `${12 / state.scale}px sans-serif`;
      ctx.fillText(`P${idx + 1}`, pt.x + 8 / state.scale, pt.y - 8 / state.scale);
    });

    if (state.currentPoints.length >= 2) {
      ctx.beginPath();
      ctx.moveTo(state.currentPoints[0].x, state.currentPoints[0].y);
      ctx.lineTo(state.currentPoints[1].x, state.currentPoints[1].y);
      ctx.stroke();
    }

    if (state.currentPoints.length === 3) {
      ctx.beginPath();
      ctx.moveTo(state.currentPoints[1].x, state.currentPoints[1].y);
      ctx.lineTo(state.currentPoints[2].x, state.currentPoints[2].y);
      ctx.lineTo(state.currentPoints[0].x, state.currentPoints[0].y);
      ctx.stroke();

      // Closing edge back to P1. The other rubber-band edge is the ticked one
      // drawn below.
      ctx.setLineDash([4 / state.scale, 4 / state.scale]);
      ctx.strokeStyle = '#ffffff';
      ctx.beginPath();
      ctx.moveTo(state.mouseImgPos.x, state.mouseImgPos.y);
      ctx.lineTo(state.currentPoints[0].x, state.currentPoints[0].y);
      ctx.stroke();
      ctx.setLineDash([]);

      const previewPts = sortQuadPoints([...state.currentPoints, state.mouseImgPos]);
      drawQuadrilateralAndGrid({ points: previewPts }, 'rgba(255, 255, 255, 0.55)', 0.6);
    }

    // Drawn last so the ticks stay readable over the preview grid.
    if (state.currentPoints.length < 4) {
      drawTickedLine(state.currentPoints[state.currentPoints.length - 1],
        state.mouseImgPos);
    }
  }

  // 4. Draw Duplicated Shape moving with mouse
  if (state.copyingQuad) {
    const copying = state.copyingQuad;
    const offsetPts = copying.points.map(p => ({
      x: p.x + (state.mouseImgPos.x - copying.anchor.x),
      y: p.y + (state.mouseImgPos.y - copying.anchor.y)
    }));

    const previewPts = sortQuadPoints(offsetPts);
    drawQuadrilateralAndGrid({ points: previewPts }, 'rgba(255, 255, 255, 0.8)', 0.9);
  }

  ctx.restore();
}

// The solved block grid, drawn once per split rather than once per quad.
function drawPlacementLattices(): void {
  const drawn = new Set<string>();
  state.quads.forEach(quad => {
    if (!quad.placement || drawn.has(quad.placement.groupId)) return;
    drawn.add(quad.placement.groupId);
    drawPlacementLattice(quad.placement);
  });
}

// The detected block edges plus one block of grid on every side, so the lattice can
// be checked against blocks the selection never covered.
function drawPlacementLattice(placement: Placement): void {
  const H = computeHomography(placement.lattice.pts);
  if (!H) return;
  const { nu, nv } = placement.lattice;
  const su = 1 / nu, sv = 1 / nv;
  const du = 0, dv = 0;

  ctx.save();

  // A homography maps a straight line to a straight line, so two endpoints are
  // enough per lattice line however far it is extended.
  const line = (u1: number, v1: number, u2: number, v2: number) => {
    const a = transformHomography(H, u1, v1);
    const b = transformHomography(H, u2, v2);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  };

  // Context ring first, faint and dashed; the detected edges go over it solid.
  for (const ring of [true, false]) {
    const iLo = ring ? -1 : 0, iHi = ring ? nu + 1 : nu;
    const jLo = ring ? -1 : 0, jHi = ring ? nv + 1 : nv;
    const uLo = du + iLo * su, uHi = du + iHi * su;
    const vLo = dv + jLo * sv, vHi = dv + jHi * sv;

    ctx.strokeStyle = ring ? 'rgba(255, 255, 255, 0.35)' : ACCENT;
    ctx.lineWidth = (ring ? 1 : 2) / state.scale;
    ctx.setLineDash(ring ? [4 / state.scale, 4 / state.scale] : []);

    for (let i = iLo; i <= iHi; i++) line(du + i * su, vLo, du + i * su, vHi);
    for (let j = jLo; j <= jHi; j++) line(uLo, dv + j * sv, uHi, dv + j * sv);
  }

  ctx.restore();
}

// Rubber-band edge from the last placed point to the cursor, ticked into the
// same 16 divisions the quad will be sampled at, so an edge can be lined up
// with block boundaries while it is still being drawn. Every 4th tick is
// longer, which keeps the quarters countable when 16 marks get dense.
function drawTickedLine(from: Point, to: Point, divisions = 16): void {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const len = Math.hypot(dx, dy);

  ctx.save();
  ctx.strokeStyle = '#ffffff';

  ctx.lineWidth = 1.5 / state.scale;
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();

  // Below a pixel or so the ticks would all land on the same spot.
  if (len > 1e-6) {
    const nx = -dy / len;   // unit normal, for ticks drawn across the line
    const ny = dx / len;

    ctx.lineWidth = 1 / state.scale;
    ctx.beginPath();
    for (let i = 1; i < divisions; i++) {
      const t = i / divisions;
      const px = from.x + dx * t;
      const py = from.y + dy * t;
      const h = (i % 4 === 0 ? 7 : 4) / state.scale;
      ctx.moveTo(px - nx * h, py - ny * h);
      ctx.lineTo(px + nx * h, py + ny * h);
    }
    ctx.stroke();
  }

  ctx.restore();
}

// Render Quad, 16x16 Grid & North Compass Indicator directly on canvas
function drawQuadrilateralAndGrid(quad: DrawableQuad, color: string, alpha: number): void {
  const quadPts = quad.points;
  if (!quadPts || quadPts.length !== 4) return;

  ctx.save();
  ctx.globalAlpha = alpha;

  // Tint inside quad
  ctx.beginPath();
  ctx.moveTo(quadPts[0].x, quadPts[0].y);
  ctx.lineTo(quadPts[1].x, quadPts[1].y);
  ctx.lineTo(quadPts[2].x, quadPts[2].y);
  ctx.lineTo(quadPts[3].x, quadPts[3].y);
  ctx.closePath();
  ctx.fillStyle = 'rgba(224, 64, 44, 0.07)';
  ctx.fill();

  // Outer quad border
  ctx.strokeStyle = color;
  ctx.lineWidth = 2 / state.scale;
  ctx.stroke();

  // 16x16 Grid Lines using Homography / Bilinear
  ctx.lineWidth = 1 / state.scale;
  ctx.strokeStyle = color;
  ctx.globalAlpha = alpha * 0.45;

  // Vertical lines (i = 1 .. 15)
  for (let i = 1; i < 16; i++) {
    const u = i / 16;
    const pTop = evaluateQuadPoint(quadPts, u, 0);
    const pBottom = evaluateQuadPoint(quadPts, u, 1);

    ctx.beginPath();
    ctx.moveTo(pTop.x, pTop.y);
    ctx.lineTo(pBottom.x, pBottom.y);
    ctx.stroke();
  }

  // Horizontal lines (j = 1 .. 15)
  for (let j = 1; j < 16; j++) {
    const v = j / 16;
    const pLeft = evaluateQuadPoint(quadPts, 0, v);
    const pRight = evaluateQuadPoint(quadPts, 1, v);

    ctx.beginPath();
    ctx.moveTo(pLeft.x, pLeft.y);
    ctx.lineTo(pRight.x, pRight.y);
    ctx.stroke();
  }

  // Corner handles & points
  ctx.globalAlpha = alpha;
  quadPts.forEach((pt, idx) => {
    const isHovered = state.hoveredHandle && quad.id && state.hoveredHandle.quadId === quad.id && state.hoveredHandle.pointIdx === idx;
    const isDragged = state.draggedHandle && quad.id && state.draggedHandle.quadId === quad.id && state.draggedHandle.pointIdx === idx;

    const radius = (isHovered || isDragged) ? 7 / state.scale : 4 / state.scale;

    ctx.beginPath();
    ctx.arc(pt.x, pt.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = (isHovered || isDragged) ? '#ffffff' : color;
    ctx.fill();

    ctx.lineWidth = 1.5 / state.scale;
    ctx.strokeStyle = (isHovered || isDragged) ? '#000000' : '#ffffff';
    ctx.stroke();
  });

  // Overlay North Compass Arrow directly on the Canvas Overlay using Quadrilateral Local Geometry!
  // Blocks from a placement solve show their edges instead; the facing is still
  // computed and still in the right-click menu.
  if (!quad.placement && quad.analysisResult && quad.analysisResult.analysis) {
    const angle = quad.analysisResult.analysis.angle;
    drawNorthArrowOnQuadCanvas(ctx, quadPts, angle);

    // Label orientation badge near top-left vertex
    const ori = quad.analysisResult.analysis.bestOrientation;
    ctx.fillStyle = '#ffffff';
    ctx.font = `bold ${12 / state.scale}px sans-serif`;
    ctx.fillText(ori, quadPts[0].x, quadPts[0].y - 8 / state.scale);
  }

  ctx.restore();
}
