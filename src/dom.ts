// --- DOM Elements ---
// Looked up once at load. `el` throws on a missing id rather than handing back null:
// with the old `getElementById` a single typo produced a null that only surfaced at the
// first `addEventListener`, taking the rest of the script down with it and leaving a
// half-wired page. Failing here names the element that is missing.
function el<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`missing element #${id}`);
  return found as T;
}

export const viewport = el<HTMLDivElement>('viewport');
export const canvas = el<HTMLCanvasElement>('mainCanvas');

const context = canvas.getContext('2d');
if (!context) throw new Error('2d canvas context unavailable');
export const ctx: CanvasRenderingContext2D = context;

export const fileInput = el<HTMLInputElement>('fileInput');
export const importJsonInput = el<HTMLInputElement>('importJsonInput');
export const uploadOverlay = el<HTMLDivElement>('uploadOverlay');
export const uploadBox = el<HTMLDivElement>('uploadBox');
export const statusBadge = el<HTMLDivElement>('statusBadge');

// Context Menus
export const quadContextMenu = el<HTMLDivElement>('quadContextMenu');
export const globalContextMenu = el<HTMLDivElement>('globalContextMenu');

export const menuPreviewCanvas = el<HTMLCanvasElement>('menuPreviewCanvas');
export const menuOrientationBadge = el<HTMLDivElement>('menuOrientationBadge');
export const menuConfidenceBadge = el<HTMLDivElement>('menuConfidenceBadge');
export const menuScoresGrid = el<HTMLDivElement>('menuScoresGrid');
export const menuPlacement = el<HTMLDivElement>('menuPlacement');
export const menuUndoPlacement = el<HTMLDivElement>('menuUndoPlacement');
export const menuDownload = el<HTMLDivElement>('menuDownload');
export const menuCopy = el<HTMLDivElement>('menuCopy');
export const menuRemove = el<HTMLDivElement>('menuRemove');

export const gridModeSelect = el<HTMLSelectElement>('gridModeSelect');
export const northGuideSelect = el<HTMLSelectElement>('northGuideSelect');
export const globalExportJson = el<HTMLDivElement>('globalExportJson');
export const globalResetView = el<HTMLDivElement>('globalResetView');
export const globalClearShapes = el<HTMLDivElement>('globalClearShapes');
export const globalLoadImgLabel = el<HTMLLabelElement>('globalLoadImgLabel');
export const globalImportJsonLabel = el<HTMLLabelElement>('globalImportJsonLabel');
export const modKeyHint = el<HTMLElement>('modKeyHint');
