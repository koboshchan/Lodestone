// --- Block Placement Solver ---
// A screenshot of a grass patch shows no seams, so the block grid has to be
// recovered rather than eyeballed. The top texture is a known 16x16 image at one
// of four rotations, which makes a candidate grid directly scorable: resample each
// cell to 16x16 and correlate it against the reference. Only the true grid makes
// every cell look like the reference.
//
// Unlike analyzeOrientation this correlates the *whole* tile rather than the
// centre 6x6. The centre is used there because it tolerates misalignment; here the
// misalignment is exactly what is being measured, and it shows up at the edges.
//
// This module holds the fixed parts: the constants, the prepared reference, and the
// image luminance buffer. It imports nothing but `constants` and `state`, so its
// top-level work (building REFERENCE_LEVELS) cannot be caught in an import cycle.

import { REFERENCE_GRASS_BLOCK_TOP } from '../constants.js';
import { state } from '../state.js';

export const TILE = 16;
export const TILE_N = TILE * TILE;

// Below this mean correlation the fit is rejected rather than reported. A true
// alignment scores near 1, so this is not a tight bound - it is set well above the
// ~0.35 that unrelated noise can reach by chance over a single cell. Expect to land
// here when the screenshot has fewer than ~16 pixels per block: there is no texel
// detail left to match, and a wrong grid is worse than no grid.
export const PLACEMENT_MIN_SCORE = 0.45;

// How far cell size may stray from "the selection bounds the blocks". Enough to
// absorb a roughly drawn box, not enough for a cell to shrink onto part of a block.
export const PLACEMENT_SCALE_TOL = 0.25;

export interface LuminanceBuffer {
  image: HTMLImageElement | null;
  data: Float32Array;
  w: number;
  h: number;
}

// Luminance of the loaded image, built once per image. A single solve reads a few
// hundred thousand pixels; re-uploading the image the way updateQuadAnalysis does
// would dominate the search.
let luminanceCache: LuminanceBuffer = { image: null, data: new Float32Array(0), w: 0, h: 0 };

export function getImageLuminance(): LuminanceBuffer {
  if (luminanceCache.image === state.image) return luminanceCache;
  if (!state.image) return luminanceCache;

  const w = state.image.width;
  const h = state.image.height;
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const cx = c.getContext('2d');
  if (!cx) return luminanceCache;
  cx.drawImage(state.image, 0, 0);
  const src = cx.getImageData(0, 0, w, h).data;

  const data = new Float32Array(w * h);
  for (let i = 0, p = 0; i < data.length; i++, p += 4) {
    data[i] = 0.299 * src[p] + 0.587 * src[p + 1] + 0.114 * src[p + 2];
  }

  luminanceCache = { image: state.image, data, w, h };
  return luminanceCache;
}

// Comparison resolutions, coarse to fine. A cell is always sampled at 16x16 and
// then box-averaged down to the level's size. Averaging is what makes the search
// possible: at full resolution the objective is nearly a delta function - the
// texture is high-frequency noise, so one source pixel of drift takes the
// correlation from 1.0 to nothing, and there is no slope to follow. Each 2x
// reduction roughly doubles the basin of attraction.
export const PLACEMENT_LEVELS = [4, 8, 16];

// Removes the mean and any linear gradient from an n x n tile, then scales it to
// unit deviation so scoring is a plain dot product. The gradient matters:
// Minecraft shades a top face with ambient occlusion, a smooth ramp across the
// block that the flat reference has no trace of. Left in, it swamps the texture
// detail the fit depends on. Returns false for a tile with no detail left, which
// cannot be matched.
export function deplaneNormalize(tile: Float32Array, n: number): boolean {
  const mid = (n - 1) / 2;
  // The sample grid is centred and symmetric, so sum(x) = sum(y) = sum(xy) = 0 and
  // the least-squares normal equations collapse into three independent sums.
  const gxx = n * n * (n * n - 1) / 12;   // sum over the tile of (c - mid)^2
  let sum = 0, sumX = 0, sumY = 0;
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const v = tile[r * n + c];
      sum += v;
      sumX += v * (c - mid);
      sumY += v * (r - mid);
    }
  }
  const mean = sum / (n * n);
  const gx = sumX / gxx;
  const gy = sumY / gxx;

  let sq = 0;
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const i = r * n + c;
      const v = tile[i] - mean - gx * (c - mid) - gy * (r - mid);
      tile[i] = v;
      sq += v * v;
    }
  }

  const sd = Math.sqrt(sq / (n * n));
  if (sd < 1e-4) return false;
  for (let i = 0; i < n * n; i++) tile[i] /= sd;
  return true;
}

// Box-averages a 16x16 tile down to n x n in place-compatible fashion.
export function reduceTile(tile: Float32Array, n: number, out: Float32Array): void {
  if (n === TILE) {
    out.set(tile);
    return;
  }
  const k = TILE / n;
  const inv = 1 / (k * k);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      let sum = 0;
      for (let dr = 0; dr < k; dr++) {
        const row = (r * k + dr) * TILE + c * k;
        for (let dc = 0; dc < k; dc++) sum += tile[row + dc];
      }
      out[r * n + c] = sum * inv;
    }
  }
}

// The reference at each rotation, prepared once per level. Same index maths as
// getCenter6x6FromRef, over the full tile.
const REFERENCE_ROTATIONS = [0, 90, 180, 270].map(angle => {
  const tile = new Float32Array(TILE_N);
  for (let r = 0; r < TILE; r++) {
    for (let c = 0; c < TILE; c++) {
      let origR: number, origC: number;
      if (angle === 0) {
        origR = r; origC = c;
      } else if (angle === 90) {
        origR = 15 - c; origC = r;
      } else if (angle === 180) {
        origR = 15 - r; origC = 15 - c;
      } else {
        origR = c; origC = 15 - r;
      }
      tile[r * TILE + c] = REFERENCE_GRASS_BLOCK_TOP[origR][origC];
    }
  }
  return tile;
});

export const REFERENCE_LEVELS = PLACEMENT_LEVELS.map(n =>
  REFERENCE_ROTATIONS.map(full => {
    const tile = new Float32Array(n * n);
    reduceTile(full, n, tile);
    deplaneNormalize(tile, n);
    return tile;
  }));
