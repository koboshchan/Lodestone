// Shared shapes. These were all implicit before the TypeScript conversion — built in
// one place and destructured in several others — which is exactly where the conversion
// pays for itself.

export interface Point {
  x: number;
  y: number;
}

/** Corner order is TL, TR, BR, BL — see `sortQuadPoints`. */
export type QuadPoints = [Point, Point, Point, Point];

export type GridMode = 'homography' | 'bilinear';

/**
 * A homography as a flat record rather than a matrix. Deliberate: the solver reads
 * these fields directly in its innermost loop, once per texel, and neither array
 * indexing nor a wrapper object survives that budget.
 */
export interface Homography {
  h00: number; h01: number; h02: number;
  h10: number; h11: number; h12: number;
  h20: number; h21: number;
}

export interface OrientationScore {
  name: string;
  code: string;
  facing: number;
  rawAngle: number;
  corr: number;
  angle: number;
}

export interface Orientation {
  bestOrientation: string;
  facing: number;
  angle: number;
  confidence: number;
  bestCorr: number;
  scores: OrientationScore[];
}

export interface AnalysisResult {
  dataUrl: string;
  analysis: Orientation;
  canvas: HTMLCanvasElement;
}

/**
 * A candidate block grid. `pts` are its own four image-space corners rather than an
 * offset within the drawn selection: a hand-drawn quad's corners are each wrong by a
 * different amount, so the true lattice is a full projective transform of it, not a
 * shift and a stretch.
 *
 * `refCell*` and `refC*` are the plausibility anchor — the cell size and centre the
 * user's selection implied — and never change as a candidate is refined.
 */
export interface Lattice {
  nu: number;
  nv: number;
  refCellU: number;
  refCellV: number;
  refCx: number;
  refCy: number;
  pts: Point[];
  score: number;
}

export interface LatticeCell {
  iu: number;
  iv: number;
  corr: number;
  rotation: number;
}

export interface PlacementSolution {
  lattice: Lattice;
  cells: LatticeCell[];
}

export interface Placement {
  groupId: string;
  iu: number;
  iv: number;
  corr: number;
  /** The selection this group replaced, kept so `undoPlacement` can restore it. */
  frame: Point[];
  lattice: Lattice;
}

export interface Quad {
  id: number;
  points: Point[];
  analysisResult?: AnalysisResult;
  placement?: Placement;
}

/**
 * A duplicate being carried on the cursor. Not a `Quad`: it has no id and never enters
 * `state.quads` — it becomes one only when clicked into place.
 */
export interface CopyingQuad {
  points: Point[];
  anchor: Point;
}

/**
 * What the renderer needs to draw an outline and grid. Loose because previews (the
 * rubber-band shape, the duplicate on the cursor) are drawn before they are real quads.
 */
export interface DrawableQuad {
  points: Point[];
  id?: number;
  analysisResult?: AnalysisResult;
  placement?: Placement;
}

/**
 * A quad corner being dragged or hovered. Identified by quad *id* rather than index,
 * so a drag survives the array being spliced underneath it — which a placement split
 * does.
 */
export interface HandleRef {
  quadId: number;
  pointIdx: number;
}

export interface AppState {
  image: HTMLImageElement | null;
  imageDataUrl: string | null;
  imageLoaded: boolean;
  gridMode: GridMode;
  northGuideAngle: number;
  scale: number;
  panX: number;
  panY: number;
  isPanning: boolean;
  panStartX: number;
  panStartY: number;
  isSpacePressed: boolean;
  quads: Quad[];
  currentPoints: Point[];
  mouseImgPos: Point;
  copyingQuad: CopyingQuad | null;
  draggedHandle: HandleRef | null;
  hoveredHandle: HandleRef | null;
  contextQuadId: number | null;
}

/** The on-disk / localStorage shape. Older files have no `placement` on their quads. */
export interface ProjectFile {
  appName?: string;
  version?: string;
  exportedAt?: string;
  gridMode?: GridMode;
  northGuideAngle?: number;
  imageDataUrl?: string | null;
  quads?: Quad[];
}
