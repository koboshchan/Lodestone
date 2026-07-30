import type { AppState } from './types.js';

// --- Application State ---
// One mutable singleton, as before. Every module reads and writes it directly; making it
// a store with accessors would add ceremony without buying anything, since there is
// exactly one canvas and one document.
export const state: AppState = {
  image: null,
  imageDataUrl: null,
  imageLoaded: false,
  gridMode: 'homography',
  northGuideAngle: 0,
  scale: 1,
  panX: 0,
  panY: 0,
  isPanning: false,
  panStartX: 0,
  panStartY: 0,
  isSpacePressed: false,
  quads: [],
  currentPoints: [],
  mouseImgPos: { x: 0, y: 0 },
  copyingQuad: null,
  draggedHandle: null,
  hoveredHandle: null,
  contextQuadId: null
};
