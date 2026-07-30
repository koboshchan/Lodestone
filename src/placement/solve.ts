import { state } from '../state.js';
import { computeHomography, sortQuadPoints, transformHomography } from '../geometry.js';
import { updateQuadAnalysis } from '../orientation.js';
import type { Lattice, LatticeCell, Point, PlacementSolution, Quad } from '../types.js';
import {
  PLACEMENT_LEVELS, PLACEMENT_MIN_SCORE, PLACEMENT_SCALE_TOL, getImageLuminance,
  type LuminanceBuffer
} from './reference.js';
import {
  evaluateLattice, latticeAxes, latticeCentre, latticePlausible, placementRank
} from './score.js';

interface SearchMode {
  /** 0 = the candidate's u axis, 1 = its v axis. */
  axis: 0 | 1;
  /** How far each of the four corners moves, in units of the step. */
  w: number[];
}

// Search directions over a candidate's corners. The whole-quad moves come first
// because they take out most of the error at once; the per-corner moves after them
// clean up the shape, which is the part a translation and a stretch cannot reach.
// Each entry is the axis to move along and a weight per corner.
const PLACEMENT_MODES: SearchMode[] = (() => {
  const modes: SearchMode[] = [
    { axis: 0, w: [1, 1, 1, 1] },      // translate along u
    { axis: 1, w: [1, 1, 1, 1] },      // translate along v
    { axis: 0, w: [-1, 1, 1, -1] },    // stretch along u
    { axis: 1, w: [-1, -1, 1, 1] }     // stretch along v
  ];
  for (let i = 0; i < 4; i++) {
    for (const axis of [0, 1] as const) {
      const w = [0, 0, 0, 0];
      w[i] = 1;
      modes.push({ axis, w });
    }
  }
  return modes;
})();

// Coordinate pattern search with a halving step, in pixels along the candidate's
// own axes.
function refineLattice(
  lum: LuminanceBuffer, lat: Lattice, level: number, startFrac: number, passes: number
): Lattice {
  let best: Lattice = Object.assign({}, lat);
  best.score = evaluateLattice(lum, best, level, null);

  const ax = latticeAxes(best.pts);
  if (!ax) return best;
  let step = startFrac * Math.min(ax.ul / best.nu, ax.vl / best.nv);

  for (let pass = 0; pass < passes; pass++) {
    // Sweep at this step size until it stops paying, so a corner can travel several
    // steps in one pass rather than one step per halving.
    for (let sweep = 0; sweep < 8; sweep++) {
      let improved = false;
      const axes = latticeAxes(best.pts);
      if (!axes) break;

      for (const mode of PLACEMENT_MODES) {
        const dx = (mode.axis === 0 ? axes.ux : axes.vx) * step;
        const dy = (mode.axis === 0 ? axes.uy : axes.vy) * step;

        for (const dir of [1, -1]) {
          const trial: Lattice = Object.assign({}, best);
          trial.pts = best.pts.map((p, i) => ({
            x: p.x + dir * dx * mode.w[i],
            y: p.y + dir * dy * mode.w[i]
          }));
          if (!latticePlausible(trial)) continue;
          const score = evaluateLattice(lum, trial, level, null);
          if (score > best.score) {
            trial.score = score;
            best = trial;
            improved = true;
          }
        }
      }
      if (!improved) break;
    }
    step /= 2;
  }
  return best;
}

// The drawn quad gives a homography from the unit square to the image, which is
// used to lay out candidate lattices: a block grid on a flat surface is projective
// too, so sliding and stretching within that frame enumerates sensible starting
// guesses. The homography stays valid outside the unit square, which is what lets a
// candidate hang off the edges of the selection. From there each candidate is
// refined on its own corners, since the drawn frame is only approximately right.
//
// Always solved in the homography frame regardless of state.gridMode: the bilinear
// map has no meaningful extension past its corners.
function solvePlacementPass(framePoints: Point[], anchor: Point): Lattice | null {
  const frame = computeHomography(framePoints);
  if (!frame) return null;
  const lum = getImageLuminance();
  const corners: [number, number][] = [[0, 0], [1, 0], [1, 1], [0, 1]];

  // Coarse pass, on the most averaged level: the lattice offset swept over half a
  // period each way, at three cell sizes spanning the tolerance. A whole-period
  // shift is deliberately not searched - that lattice sits on blocks the selection
  // does not cover - so the cost is that the box has to be drawn within half a block
  // of right. The three sizes keep the worst-case size error inside this level's
  // basin; one nominal size is not enough, and a box drawn 20% too big is ordinary.
  const COARSE_STEPS = 11;
  const COARSE_SCALES = [1 - PLACEMENT_SCALE_TOL * 0.6, 1, 1 + PLACEMENT_SCALE_TOL * 0.6];
  const COARSE_KEEP = 12;
  const candidates: Lattice[] = [];
  const ref = latticeAxes(corners.map(([a, b]) => transformHomography(frame, a, b)));
  if (!ref) return null;

  for (const nu of [1, 2]) {
    for (const nv of [1, 2]) {
      const seeds: Lattice[] = [];

      for (const fu of COARSE_SCALES) {
        for (const fv of COARSE_SCALES) {
          for (let i = 0; i < COARSE_STEPS; i++) {
            for (let j = 0; j < COARSE_STEPS; j++) {
              const du = (i / (COARSE_STEPS - 1) - 0.5) * (fu / nu);
              const dv = (j / (COARSE_STEPS - 1) - 0.5) * (fv / nv);
              const lat: Lattice = {
                nu, nv,
                refCellU: ref.ul / nu, refCellV: ref.vl / nv,
                refCx: anchor.x, refCy: anchor.y,
                pts: corners.map(([a, b]) =>
                  transformHomography(frame, du + a * fu, dv + b * fv)),
                score: 0
              };
              if (!latticePlausible(lat)) continue;
              lat.score = evaluateLattice(lum, lat, 0, null);
              if (lat.score > 0) seeds.push(lat);
            }
          }
        }
      }

      // Kept per arrangement, not globally. A 16-value comparison throws up
      // spurious 0.6s readily, and one noisy arrangement would otherwise fill the
      // shortlist and starve the others.
      seeds.sort((a, b) => b.score - a.score);
      candidates.push(...seeds.slice(0, COARSE_KEEP));
    }
  }
  if (!candidates.length) return null;

  // Down the ladder, pruning as it goes. A level's score shortlists but does not
  // decide - it has averaged away the detail needed for that - so several survivors
  // are carried to the next level and only the full-resolution fit picks a winner.
  // Pruning is what makes the widest search affordable at the cheapest level, which
  // is where it is needed: that is the level whose basin is wide enough to reach a
  // sloppily drawn selection in the first place.
  const LADDER_KEEP = [12, 6];
  const STEP_FRACTION = [1 / 8, 1 / 16, 1 / 32];
  let pool = candidates;
  for (let level = 0; level < PLACEMENT_LEVELS.length; level++) {
    pool = pool.map(lat => refineLattice(lum, lat, level, STEP_FRACTION[level], 5));
    pool.sort((a, b) => placementRank(b) - placementRank(a));
    if (level < LADDER_KEEP.length) pool = pool.slice(0, LADDER_KEEP[level]);
  }

  return pool[0] || null;
}

// A pass corrects position and size well, but can still be left short by shape: the
// four corners of a hand-drawn selection are each wrong by a different amount, and
// that error can start outside even the coarsest level's basin, where there is no
// slope to follow. Re-running from what the previous pass found fixes it, because
// that frame already has close to the right shape. The drift limit stays pinned to
// the selection the user actually drew so the lattice cannot walk a block per pass.
export function solvePlacement(quad: Quad): PlacementSolution | null {
  if (!state.image || !quad || !quad.points || quad.points.length !== 4) return null;

  const anchor = latticeCentre(quad.points);
  let best = solvePlacementPass(quad.points, anchor);
  if (!best) return null;

  for (let pass = 1; pass < 3 && best.score < 0.999; pass++) {
    const next = solvePlacementPass(best.pts, anchor);
    if (!next || next.score <= best.score + 1e-4) break;
    best = next;
  }

  if (best.score < PLACEMENT_MIN_SCORE) return null;

  const cells: LatticeCell[] = [];
  if (evaluateLattice(getImageLuminance(), best, PLACEMENT_LEVELS.length - 1, cells) < 0) {
    return null;
  }
  return { lattice: best, cells };
}

// Replaces the rough selection with one quad per detected block, each aligned to
// the solved lattice and analyzed like any other quad.
export function applyPlacement(quad: Quad, solution: PlacementSolution): Quad[] {
  const { lattice, cells } = solution;
  const groupId = `g${Date.now()}`;
  const frame: Point[] = JSON.parse(JSON.stringify(quad.points));
  const corners: [number, number][] = [[0, 0], [1, 0], [1, 1], [0, 1]];

  const H = computeHomography(lattice.pts);
  const su = 1 / lattice.nu, sv = 1 / lattice.nv;

  const created = cells.map((cell, n) => {
    const u0 = cell.iu * su;
    const v0 = cell.iv * sv;
    const pts = corners.map(([a, b]) => transformHomography(
      H, u0 + a * su, v0 + b * sv));

    const newQuad: Quad = {
      // Date.now() alone collides here: every cell of a split is created inside the
      // same millisecond.
      id: Date.now() + n,
      points: sortQuadPoints(pts),
      placement: { groupId, iu: cell.iu, iv: cell.iv, corr: cell.corr, frame, lattice }
    };
    updateQuadAnalysis(newQuad);
    return newQuad;
  });

  const idx = state.quads.findIndex(q => q.id === quad.id);
  if (idx < 0) state.quads.push(...created);
  else state.quads.splice(idx, 1, ...created);
  return created;
}

// The split is destructive, so it has a way back: drop the whole group and put the
// original selection back where it was.
export function undoPlacement(groupId: string): boolean {
  const idx = state.quads.findIndex(q => q.placement && q.placement.groupId === groupId);
  if (idx < 0) return false;

  const target = state.quads[idx].placement;
  if (!target) return false;

  const restored: Quad = {
    id: Date.now(),
    points: JSON.parse(JSON.stringify(target.frame))
  };
  updateQuadAnalysis(restored);

  state.quads = state.quads.filter(q => !(q.placement && q.placement.groupId === groupId));
  state.quads.splice(Math.min(idx, state.quads.length), 0, restored);
  return true;
}
