import { computeHomography } from '../geometry.js';
import type { Homography, Lattice, LatticeCell, Point } from '../types.js';
import {
  PLACEMENT_LEVELS, PLACEMENT_SCALE_TOL, REFERENCE_LEVELS, TILE, TILE_N,
  deplaneNormalize, reduceTile,
  type LuminanceBuffer
} from './reference.js';

// Scratch buffers, reused across the thousands of candidates a solve evaluates.
const placementTile = new Float32Array(TILE_N);
const placementReduced = new Float32Array(TILE_N);

// Resamples one lattice cell to a 16x16 luminance tile. False if the cell leaves
// the image, which rejects the candidate it belongs to.
export function sampleCellTile(
  H: Homography, lum: LuminanceBuffer,
  u0: number, v0: number, su: number, sv: number,
  out: Float32Array
): boolean {
  for (let r = 0; r < TILE; r++) {
    const v = v0 + (r + 0.5) / TILE * sv;
    for (let c = 0; c < TILE; c++) {
      const u = u0 + (c + 0.5) / TILE * su;
      const w = H.h20 * u + H.h21 * v + 1;
      if (Math.abs(w) < 1e-9) return false;   // on the horizon of the frame
      const px = Math.floor((H.h00 * u + H.h01 * v + H.h02) / w);
      const py = Math.floor((H.h10 * u + H.h11 * v + H.h12) / w);
      if (px < 0 || px >= lum.w || py < 0 || py >= lum.h) return false;
      out[r * TILE + c] = lum.data[py * lum.w + px];
    }
  }
  return true;
}

// Best correlation of a tile over the four rotations, at one level of the ladder.
// Writes the winning rotation index to lastTileRotation rather than allocating a
// result object per candidate.
let lastTileRotation = 0;

export function scoreTile(tile: Float32Array, level: number): number {
  const n = PLACEMENT_LEVELS[level];
  reduceTile(tile, n, placementReduced);
  if (!deplaneNormalize(placementReduced, n)) return -1;

  const refs = REFERENCE_LEVELS[level];
  const count = n * n;
  let best = -1;
  for (let k = 0; k < 4; k++) {
    const ref = refs[k];
    let dot = 0;
    for (let i = 0; i < count; i++) dot += placementReduced[i] * ref[i];
    dot /= count;
    if (dot > best) {
      best = dot;
      lastTileRotation = k;
    }
  }
  return best;
}

// A candidate is the image-space quadrilateral that its nu x nv block region
// occupies, in the usual TL,TR,BR,BL order. Cell (iu,iv) is the sub-rectangle
// [iu/nu,(iu+1)/nu] x [iv/nv,(iv+1)/nv] of that quad's own unit square.
//
// The candidate carries its own corners rather than an offset and a size within the
// drawn selection's frame. It has to: the four points of a hand-drawn selection are
// each wrong by a different amount, so the true lattice sits at a full projective
// transform of the drawn frame, not a translation and a stretch of it.
export function evaluateLattice(
  lum: LuminanceBuffer, lat: Lattice, level: number, cellsOut: LatticeCell[] | null
): number {
  const H = computeHomography(lat.pts);
  if (!H) return -1;
  const su = 1 / lat.nu, sv = 1 / lat.nv;

  let total = 0;
  for (let iv = 0; iv < lat.nv; iv++) {
    for (let iu = 0; iu < lat.nu; iu++) {
      if (!sampleCellTile(H, lum, iu * su, iv * sv, su, sv, placementTile)) return -1;
      const corr = scoreTile(placementTile, level);
      if (corr < -0.5) return -1;
      if (cellsOut) cellsOut.push({ iu, iv, corr, rotation: lastTileRotation });
      total += corr;
    }
  }
  return total / (lat.nu * lat.nv);
}

export interface LatticeAxes {
  ux: number; uy: number;
  vx: number; vy: number;
  ul: number; vl: number;
}

// Unit vectors along the candidate's own axes, so a step means the same thing
// whatever angle the selection sits at on screen.
export function latticeAxes(pts: Point[]): LatticeAxes | null {
  const ux = ((pts[1].x - pts[0].x) + (pts[2].x - pts[3].x)) / 2;
  const uy = ((pts[1].y - pts[0].y) + (pts[2].y - pts[3].y)) / 2;
  const vx = ((pts[3].x - pts[0].x) + (pts[2].x - pts[1].x)) / 2;
  const vy = ((pts[3].y - pts[0].y) + (pts[2].y - pts[1].y)) / 2;
  const ul = Math.hypot(ux, uy), vl = Math.hypot(vx, vy);
  if (ul < 1e-6 || vl < 1e-6) return null;
  return { ux: ux / ul, uy: uy / ul, vx: vx / vl, vy: vy / vl, ul, vl };
}

export function latticeCentre(pts: Point[]): Point {
  return {
    x: (pts[0].x + pts[1].x + pts[2].x + pts[3].x) / 4,
    y: (pts[0].y + pts[1].y + pts[2].y + pts[3].y) / 4
  };
}

// How far a candidate has walked from where the selection was drawn, in blocks.
export function latticeDrift(lat: Lattice): number {
  const c = latticeCentre(lat.pts);
  return Math.hypot(c.x - lat.refCx, c.y - lat.refCy) /
    ((lat.refCellU + lat.refCellV) / 2);
}

// Cell size and position must both stay near what the selection implies, or a
// candidate could walk off onto blocks the user never pointed at.
export function latticePlausible(lat: Lattice): boolean {
  const ax = latticeAxes(lat.pts);
  if (!ax) return false;
  const cu = ax.ul / lat.nu / lat.refCellU;
  const cv = ax.vl / lat.nv / lat.refCellV;
  return cu > 1 - PLACEMENT_SCALE_TOL && cu < 1 + PLACEMENT_SCALE_TOL &&
    cv > 1 - PLACEMENT_SCALE_TOL && cv < 1 + PLACEMENT_SCALE_TOL &&
    latticeDrift(lat) < 0.75;
}

// Ranking, as opposed to fit. The neighbouring blocks are real blocks too, so a
// lattice shifted one block over fits exactly as well as the right one and the
// choice between them is otherwise arbitrary. Break the tie towards where the
// selection was actually drawn, gently enough that a genuinely better fit still wins.
export function placementRank(lat: Lattice): number {
  const d = latticeDrift(lat);
  return lat.score - 0.15 * d * d;
}
