import { state } from './state.js';
import { IS_MAC } from './constants.js';
import type { GridMode, HandleRef, Homography, Point } from './types.js';

export function screenToImage(screenX: number, screenY: number): Point {
  return {
    x: (screenX - state.panX) / state.scale,
    y: (screenY - state.panY) / state.scale
  };
}

export function imageToScreen(imgX: number, imgY: number): Point {
  return {
    x: imgX * state.scale + state.panX,
    y: imgY * state.scale + state.panY
  };
}

// --- Perspective & Math Helpers ---
export function computeHomography(pts: Point[] | null | undefined): Homography | null {
  if (!pts || pts.length !== 4) return null;
  const x0 = pts[0].x, y0 = pts[0].y;
  const x1 = pts[1].x, y1 = pts[1].y;
  const x2 = pts[2].x, y2 = pts[2].y;
  const x3 = pts[3].x, y3 = pts[3].y;

  const dx1 = x1 - x2, dx2 = x3 - x2;
  const sx = x0 - x1 + x2 - x3;

  const dy1 = y1 - y2, dy2 = y3 - y2;
  const sy = y0 - y1 + y2 - y3;

  // An affine quad (opposite sides parallel) has no vanishing point, so the projective
  // terms vanish; the same fallback covers a degenerate quad whose determinant collapses.
  const affine: Homography = {
    h00: x1 - x0, h01: x3 - x0, h02: x0,
    h10: y1 - y0, h11: y3 - y0, h12: y0,
    h20: 0, h21: 0
  };

  if (Math.abs(sx) < 1e-6 && Math.abs(sy) < 1e-6) return affine;

  const det = dx1 * dy2 - dx2 * dy1;
  if (Math.abs(det) < 1e-10) return affine;

  const h20 = (sx * dy2 - sy * dx2) / det;
  const h21 = (dx1 * sy - dy1 * sx) / det;

  return {
    h00: x1 - x0 + h20 * x1,
    h01: x3 - x0 + h21 * x3,
    h02: x0,
    h10: y1 - y0 + h20 * y1,
    h11: y3 - y0 + h21 * y3,
    h12: y0,
    h20,
    h21
  };
}

export function transformHomography(H: Homography | null, u: number, v: number): Point {
  if (!H) return { x: 0, y: 0 };
  const w = H.h20 * u + H.h21 * v + 1;
  return {
    x: (H.h00 * u + H.h01 * v + H.h02) / w,
    y: (H.h10 * u + H.h11 * v + H.h12) / w
  };
}

export function getBilinearPoint(quad: Point[], u: number, v: number): Point {
  const TL = quad[0], TR = quad[1], BR = quad[2], BL = quad[3];
  const x = (1 - u) * (1 - v) * TL.x + u * (1 - v) * TR.x + u * v * BR.x + (1 - u) * v * BL.x;
  const y = (1 - u) * (1 - v) * TL.y + u * (1 - v) * TR.y + u * v * BR.y + (1 - u) * v * BL.y;
  return { x, y };
}

export function evaluateQuadPoint(
  quad: Point[], u: number, v: number, mode: GridMode = state.gridMode
): Point {
  if (mode === 'homography') {
    const H = computeHomography(quad);
    return transformHomography(H, u, v);
  }
  return getBilinearPoint(quad, u, v);
}

export function isPointInQuad(pt: Point, quad: Point[]): boolean {
  let inside = false;
  for (let i = 0, j = quad.length - 1; i < quad.length; j = i++) {
    const xi = quad[i].x, yi = quad[i].y;
    const xj = quad[j].x, yj = quad[j].y;
    const intersect = ((yi > pt.y) !== (yj > pt.y)) &&
      (pt.x < (xj - xi) * (pt.y - yi) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

export function sortQuadPoints(pts: Point[]): Point[] {
  if (pts.length !== 4) return pts;
  const cx = (pts[0].x + pts[1].x + pts[2].x + pts[3].x) / 4;
  const cy = (pts[0].y + pts[1].y + pts[2].y + pts[3].y) / 4;

  const mapped = pts.map(p => ({
    x: p.x,
    y: p.y,
    angle: Math.atan2(p.y - cy, p.x - cx)
  }));

  mapped.sort((a, b) => a.angle - b.angle);

  let minSum = Infinity;
  let startIndex = 0;
  for (let i = 0; i < 4; i++) {
    const sum = mapped[i].x + mapped[i].y;
    if (sum < minSum) {
      minSum = sum;
      startIndex = i;
    }
  }

  const reordered: Point[] = [];
  for (let i = 0; i < 4; i++) {
    const idx = (startIndex + i) % 4;
    reordered.push({ x: mapped[idx].x, y: mapped[idx].y });
  }

  return reordered;
}

// Cmd on macOS, Ctrl on Windows/Linux. Deliberately not interchangeable: ctrl+click
// is the system context-menu gesture on macOS, and cmd is not a modifier elsewhere.
export function isPassThrough(e: MouseEvent | KeyboardEvent): boolean {
  return IS_MAC ? e.metaKey : e.ctrlKey;
}

/**
 * The corner of a committed quad nearest a screen position, for Shift-snapping.
 *
 * Two deliberate differences from `findCornerHandleNearScreen`, which is about grabbing:
 * this returns the *nearest* corner rather than the first hit of a top-down scan, and it
 * returns a copy. The copy matters — handing back the stored object would let two quads
 * share one `Point` and move together.
 *
 * `state.currentPoints` is not in `state.quads`, so a shape being drawn cannot snap to
 * its own corners, which is what "pre-existing" should mean.
 */
export function findNearestCornerScreen(
  screenX: number, screenY: number, maxDistPx: number, excludeQuadId?: number
): Point | null {
  let best: Point | null = null;
  let bestDist = maxDistPx;
  for (const q of state.quads) {
    if (q.id === excludeQuadId) continue;
    for (const p of q.points) {
      const s = imageToScreen(p.x, p.y);
      const dist = Math.hypot(screenX - s.x, screenY - s.y);
      if (dist <= bestDist) {
        bestDist = dist;
        best = p;
      }
    }
  }
  return best ? { x: best.x, y: best.y } : null;
}

export function findCornerHandleNearScreen(
  screenX: number, screenY: number, maxDistPx = 12
): HandleRef | null {
  for (let i = state.quads.length - 1; i >= 0; i--) {
    const q = state.quads[i];
    for (let j = 0; j < q.points.length; j++) {
      const ptScreen = imageToScreen(q.points[j].x, q.points[j].y);
      const dist = Math.hypot(screenX - ptScreen.x, screenY - ptScreen.y);
      if (dist <= maxDistPx) {
        return { quadId: q.id, pointIdx: j };
      }
    }
  }
  return null;
}
