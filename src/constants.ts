// The grass_block_top.png texture at facing 0 (north), as 16x16 luminance values.
// Everything the orientation analyser and the placement solver do is a correlation
// against this table at one of four rotations, so it is the one piece of data the
// whole application is built on.
export const REFERENCE_GRASS_BLOCK_TOP: readonly (readonly number[])[] = [
  [148, 195, 147, 134, 134, 143, 127, 140, 158, 150, 138, 135, 137, 167, 112, 141],
  [149, 140, 160, 151, 153, 128, 153, 130, 127, 169, 166, 147, 195, 140, 135, 139],
  [163, 141, 125, 132, 137, 136, 144, 144, 148, 159, 168, 173, 135, 119, 128, 182],
  [118, 133, 138, 151, 145, 153, 125, 146, 127, 131, 141, 151, 143, 139, 167, 125],
  [162, 131, 173, 131, 172, 174, 136, 172, 186, 117, 128, 167, 171, 126, 136, 141],
  [153, 152, 120, 166, 129, 150, 149, 155, 135, 149, 136, 131, 120, 136, 142, 148],
  [126, 156, 165, 144, 146, 131, 138, 127, 132, 135, 181, 172, 145, 167, 151, 143],
  [143, 135, 157, 139, 144, 124, 164, 132, 151, 173, 135, 129, 143, 160, 127, 169],
  [167, 155, 182, 170, 172, 129, 173, 156, 179, 170, 130, 146, 133, 147, 158, 126],
  [151, 130, 183, 141, 140, 152, 170, 128, 135, 145, 146, 169, 142, 179, 143, 129],
  [155, 158, 151, 165, 130, 182, 142, 147, 192, 121, 167, 131, 143, 136, 151, 143],
  [132, 173, 134, 163, 151, 132, 146, 137, 151, 146, 140, 123, 179, 127, 125, 138],
  [186, 151, 129, 138, 153, 145, 133, 122, 138, 145, 138, 125, 159, 165, 121, 151],
  [129, 168, 118, 146, 125, 153, 129, 195, 161, 147, 168, 155, 160, 136, 168, 174],
  [163, 161, 192, 123, 179, 171, 148, 147, 126, 149, 149, 135, 169, 157, 139, 121],
  [133, 150, 136, 137, 164, 152, 148, 147, 142, 191, 124, 169, 132, 144, 136, 158]
];

export const STORAGE_KEY = 'quadmatrix_app_state_v1';

export const IS_MAC = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);

// The only colour in the interface. Everything else on the canvas is white,
// and states are distinguished by dash pattern and fill rather than by hue.
export const ACCENT = '#e0402c';

// Zoom limits, shared by the wheel handler and by resetView so the view can never
// land somewhere the wheel would refuse to go.
export const MIN_SCALE = 0.05;
export const MAX_SCALE = 40;
