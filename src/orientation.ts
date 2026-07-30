import { REFERENCE_GRASS_BLOCK_TOP } from './constants.js';
import { state } from './state.js';
import { evaluateQuadPoint } from './geometry.js';
import type { Orientation, OrientationScore, Quad } from './types.js';

// --- Minecraft Block Orientation Analysis ---
export function getCenter6x6FromRef(
  refMatrix: readonly (readonly number[])[], angle: number
): number[] {
  const center: number[] = [];
  for (let r = 5; r <= 10; r++) {
    for (let c = 5; c <= 10; c++) {
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
      center.push(refMatrix[origR][origC]);
    }
  }
  return center;
}

export function analyzeOrientation(cand16x16ImageData: ImageData): Orientation {
  const candLuminance: number[][] = [];
  for (let r = 0; r < 16; r++) {
    const row: number[] = [];
    for (let c = 0; c < 16; c++) {
      const idx = (r * 16 + c) * 4;
      const R = cand16x16ImageData.data[idx];
      const G = cand16x16ImageData.data[idx + 1];
      const B = cand16x16ImageData.data[idx + 2];
      const lum = 0.299 * R + 0.587 * G + 0.114 * B;
      row.push(lum);
    }
    candLuminance.push(row);
  }

  const candCenter: number[] = [];
  for (let r = 5; r <= 10; r++) {
    for (let c = 5; c <= 10; c++) {
      candCenter.push(candLuminance[r][c]);
    }
  }

  const meanCand = candCenter.reduce((a, b) => a + b, 0) / candCenter.length;
  const stdCand = Math.sqrt(candCenter.reduce((a, b) => a + Math.pow(b - meanCand, 2), 0) / candCenter.length);

  const angles = [
    { name: 'North', code: 'N', facing: 0, rawAngle: 0 },
    { name: 'East', code: 'E', facing: 1, rawAngle: 90 },
    { name: 'South', code: 'S', facing: 2, rawAngle: 180 },
    { name: 'West', code: 'W', facing: 3, rawAngle: 270 }
  ];

  const scores: OrientationScore[] = angles.map(dir => {
    const evalAngle = (dir.rawAngle + state.northGuideAngle) % 360;
    const refCenter = getCenter6x6FromRef(REFERENCE_GRASS_BLOCK_TOP, evalAngle);
    const meanRef = refCenter.reduce((a, b) => a + b, 0) / refCenter.length;
    const stdRef = Math.sqrt(refCenter.reduce((a, b) => a + Math.pow(b - meanRef, 2), 0) / refCenter.length);

    let cov = 0;
    for (let i = 0; i < candCenter.length; i++) {
      cov += (candCenter[i] - meanCand) * (refCenter[i] - meanRef);
    }
    cov /= candCenter.length;

    const corr = (stdCand > 1e-5 && stdRef > 1e-5) ? cov / (stdCand * stdRef) : 0;
    return { ...dir, corr, angle: (dir.rawAngle + state.northGuideAngle) % 360 };
  });

  scores.sort((a, b) => b.corr - a.corr);

  const best = scores[0];
  const second = scores[1];

  let confidencePct = 0;
  if (best.corr > 0) {
    const margin = best.corr - second.corr;
    confidencePct = Math.min(100, Math.max(0, Math.round(best.corr * 75 + margin * 25)));
  }

  return {
    bestOrientation: best.name,
    facing: best.facing,
    angle: best.angle,
    confidence: confidencePct,
    bestCorr: best.corr,
    scores: scores
  };
}

// --- Automatic 16x16 Conversion & Analysis for Quadrilateral ---
export function updateQuadAnalysis(quad: Quad): void {
  if (!state.image || !quad || !quad.points || quad.points.length !== 4) return;

  const sampleCanvas = document.createElement('canvas');
  sampleCanvas.width = state.image.width;
  sampleCanvas.height = state.image.height;
  const sampleCtx = sampleCanvas.getContext('2d');
  if (!sampleCtx) return;
  sampleCtx.drawImage(state.image, 0, 0);

  const imgData = sampleCtx.getImageData(0, 0, state.image.width, state.image.height);
  const data = imgData.data;
  const imgW = state.image.width;
  const imgH = state.image.height;

  const outputCanvas = document.createElement('canvas');
  outputCanvas.width = 16;
  outputCanvas.height = 16;
  const outCtx = outputCanvas.getContext('2d');
  if (!outCtx) return;
  const outImgData = outCtx.createImageData(16, 16);

  for (let j = 0; j < 16; j++) {
    for (let i = 0; i < 16; i++) {
      const u0 = i / 16, u1 = (i + 1) / 16;
      const v0 = j / 16, v1 = (j + 1) / 16;

      let sumR = 0, sumG = 0, sumB = 0, sumA = 0, count = 0;

      for (let sv = 0.1; sv < 1.0; sv += 0.2) {
        for (let su = 0.1; su < 1.0; su += 0.2) {
          const u = u0 + su * (u1 - u0);
          const v = v0 + sv * (v1 - v0);

          const pt = evaluateQuadPoint(quad.points, u, v);
          const px = Math.floor(pt.x);
          const py = Math.floor(pt.y);

          if (px >= 0 && px < imgW && py >= 0 && py < imgH) {
            const idx = (py * imgW + px) * 4;
            sumR += data[idx];
            sumG += data[idx + 1];
            sumB += data[idx + 2];
            sumA += data[idx + 3];
            count++;
          }
        }
      }

      const outIdx = (j * 16 + i) * 4;
      if (count > 0) {
        outImgData.data[outIdx] = Math.round(sumR / count);
        outImgData.data[outIdx + 1] = Math.round(sumG / count);
        outImgData.data[outIdx + 2] = Math.round(sumB / count);
        outImgData.data[outIdx + 3] = Math.round(sumA / count);
      } else {
        outImgData.data[outIdx + 3] = 0;
      }
    }
  }

  outCtx.putImageData(outImgData, 0, 0);

  const analysis = analyzeOrientation(outImgData);
  const dataUrl = outputCanvas.toDataURL('image/png');

  quad.analysisResult = {
    dataUrl,
    analysis,
    canvas: outputCanvas
  };
}
