// --- Undo / Redo ---
//
// A stack of document states with a cursor, rather than a stack of reversible commands.
// The document is small — a handful of quads, each four points plus an optional
// placement — so snapshotting it outright is cheaper to write and impossible to get
// subtly wrong, which is the failure mode command-based undo tends to have.
//
// Checkpoints are taken from saveStateToStorage, so persisting and checkpointing are
// the same event. Those call sites were already placed at every meaningful change, and
// tying to them means a future mutation cannot silently escape history.

import { state } from './state.js';
import { updateQuadAnalysis } from './orientation.js';
import { saveStateToStorage } from './persistence.js';
import { render } from './render.js';
import { updateStatus } from './status.js';
import type { Placement, Point, Quad } from './types.js';

interface SnapshotQuad {
  id: number;
  points: Point[];
  placement?: Placement;
}

interface Snapshot {
  quads: SnapshotQuad[];
}

// Deep enough to cover a working session, shallow enough that the memory never matters:
// a snapshot is a few hundred bytes per quad, since the heavy fields are excluded.
const HISTORY_LIMIT = 100;

let entries: Snapshot[] = [];
let cursor = -1;

// Applying a snapshot writes to state and saves, which would otherwise checkpoint the
// undo itself and make the stack grow every time it is walked.
let applying = false;

// Three things are deliberately not in a snapshot:
//
//   analysisResult   derived from points and the current settings, and it holds a canvas
//                    element plus a ~1-2 kB base64 dataUrl per quad
//   imageDataUrl     megabytes; loading an image starts a new document instead
//   gridMode /
//   northGuideAngle  a decision, not an oversight - undo moves shapes and leaves
//                    settings alone. It needs no special case: changing a dropdown does
//                    not alter any quad's points or placement, so the snapshot comes out
//                    identical and the dedupe guard in commitHistory drops it.
function snapshot(): Snapshot {
  return structuredClone({
    quads: state.quads.map(q => ({ id: q.id, points: q.points, placement: q.placement }))
  });
}

function samePoints(a: Point[], b: Point[]): boolean {
  if (a.length !== b.length) return false;
  return a.every((p, i) => p.x === b[i].x && p.y === b[i].y);
}

function sameSnapshot(a: Snapshot | undefined, b: Snapshot): boolean {
  return !!a && JSON.stringify(a) === JSON.stringify(b);
}

function applySnapshot(snap: Snapshot): void {
  applying = true;
  try {
    // Keyed by id, and live: every settings change re-analyses all quads, so an
    // analysisResult held here always reflects the current settings.
    const prev = new Map(state.quads.map(q => [q.id, q]));

    state.quads = snap.quads.map(sq => {
      const quad: Quad = structuredClone(sq);
      const old = prev.get(quad.id);
      // Reuse the analysis when the geometry is unchanged. This is the expensive part:
      // updateQuadAnalysis re-reads the whole source image once per quad, so restoring
      // naively is O(quads x image pixels) and stalls visibly on a large screenshot.
      // Undo usually moves one quad, and this keeps the cost proportional to that.
      if (old && old.analysisResult && samePoints(old.points, quad.points)) {
        quad.analysisResult = old.analysisResult;
      }
      return quad;
    });

    state.quads.forEach(q => {
      if (!q.analysisResult) updateQuadAnalysis(q);
    });

    // Mid-gesture state cannot survive a jump: it refers to quads that may no longer
    // exist, and a half-drawn shape is not part of the state being restored.
    state.currentPoints = [];
    state.copyingQuad = null;
    state.draggedHandle = null;
    state.hoveredHandle = null;
    state.contextQuadId = null;

    saveStateToStorage();
  } finally {
    applying = false;
  }
  requestAnimationFrame(render);
}

/** Checkpoint the current document. Called from `saveStateToStorage`. */
export function commitHistory(): void {
  if (applying) return;

  const snap = snapshot();
  // Drops the saves that changed nothing a snapshot can see: a settings change, a drag
  // that ended where it started, a solve that was rejected.
  if (sameSnapshot(entries[cursor], snap)) return;

  entries.splice(cursor + 1);
  entries.push(snap);
  if (entries.length > HISTORY_LIMIT) entries.shift();
  cursor = entries.length - 1;
}

/**
 * Start a new document. A loaded image or an imported project is not a step you can walk
 * back into — and this is what keeps image data out of the stack.
 */
export function resetHistory(): void {
  entries = [snapshot()];
  cursor = 0;
}

export function canUndo(): boolean {
  return cursor > 0;
}

export function canRedo(): boolean {
  return cursor >= 0 && cursor < entries.length - 1;
}

export function undo(): boolean {
  if (!canUndo()) return false;
  cursor--;
  applySnapshot(entries[cursor]);
  return true;
}

export function redo(): boolean {
  if (!canRedo()) return false;
  cursor++;
  applySnapshot(entries[cursor]);
  return true;
}

/**
 * One undo or redo, as the keyboard and the menu both mean it. Lives here rather than in
 * the key handler so the two entry points cannot drift apart.
 */
export function runHistoryStep(isRedo: boolean): void {
  if (!isRedo && state.currentPoints.length > 0) {
    // Mid-draw, undo backs the corners out one at a time — the quad does not exist yet,
    // so these are not document history and are correspondingly not redoable.
    state.currentPoints.pop();
    updateStatus(`Point ${state.currentPoints.length}/4`);
    requestAnimationFrame(render);
    return;
  }

  const moved = isRedo ? redo() : undo();
  updateStatus(moved ? (isRedo ? 'Redo' : 'Undo')
                     : `Nothing to ${isRedo ? 'redo' : 'undo'}`);
}
