/**
 * Virtual Tryon – Virtual Jewelry Try-On
 * main.js  –  MediaPipe Hands + FaceMesh + Canvas 2D rendering
 *
 * Architecture
 * ────────────
 * 1. JewelryAssets   – loads and caches all jewelry images
 * 2. HandTracker     – wraps MediaPipe Hands
 * 3. FaceTracker     – wraps MediaPipe FaceMesh
 * 4. Renderer        – draws video frame + jewelry onto output canvas
 * 5. VTOApp          – orchestrates everything, handles UI
 */

'use strict';

// ─── Constants ───────────────────────────────────────────────────────────────

/** Landmark indices for MediaPipe Hands */
const LM = {
  WRIST: 0,
  INDEX_MCP: 5,
  MIDDLE_MCP: 9,
  RING_MCP: 13,
  RING_PIP: 14,
  RING_DIP: 15,
  PINKY_MCP: 17,
};

/** Landmark indices for MediaPipe FaceMesh (468-point model) */
const FM = {
  CHIN: 152,
  LEFT_JAW: 234,
  RIGHT_JAW: 454,
  LEFT_NECK: 172,
  RIGHT_NECK: 397,
  FOREHEAD: 10,
};

/** Camera constraints */
const CAM_WIDTH = 1280;
const CAM_HEIGHT = 720;

// ─── Geometry utilities (ported from Python transform_utils.py) ──────────────

/** Euclidean distance between two {x,y} pixel points */
function dist2D(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/** Unit vector from pixel point a → b */
function unitVec2D(a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const n = Math.hypot(dx, dy);
  return n > 1e-6 ? { x: dx / n, y: dy / n } : { x: 1, y: 0 };
}

/**
 * Convert normalised landmark → DISPLAY pixel coords.
 * MediaPipe X is in un-mirrored video space; display is mirrored,
 * so flip X: displayX = (1 - lm.x) * W
 */
function px(lm, W, H) {
  return { x: (1 - lm.x) * W, y: lm.y * H };
}

/** Convert normalised landmark → un-mirrored pixel (for depth/world math) */
function pxRaw(lm, W, H) {
  return { x: lm.x * W, y: lm.y * H };
}

/** Clamp helper */
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// ─── Angle-wrap-safe EMA smoother (ported from JewelrySmoother) ───────────────
// Python alpha=0.68 → fast-response, low-jitter
// Using alpha=0.15 → much smoother tracking with less shaking, slight latency increase
const EMA_ALPHA = 0.15;

class AngleSmoother {
  constructor(alpha = EMA_ALPHA) {
    this.alpha = alpha;
    this._cos = null; this._sin = null;
  }
  smooth(angleDeg) {
    const c = Math.cos(angleDeg * Math.PI / 180);
    const s = Math.sin(angleDeg * Math.PI / 180);
    if (this._cos === null) { this._cos = c; this._sin = s; }
    else {
      this._cos = this.alpha * c + (1 - this.alpha) * this._cos;
      this._sin = this.alpha * s + (1 - this.alpha) * this._sin;
    }
    return Math.atan2(this._sin, this._cos) * 180 / Math.PI;
  }
  reset() { this._cos = this._sin = null; }
}

class JewelrySmoother {
  constructor(alpha = EMA_ALPHA, angleAlpha = 0.05) {
    this.alpha = alpha;
    this.angleAlpha = angleAlpha;
    this._state = {};
    this._angleSm = new AngleSmoother(angleAlpha);
  }
  smooth({ cx, cy, scale, angle }) {
    const lerp = (key, val) => {
      if (!(key in this._state)) this._state[key] = val;
      else this._state[key] = this.alpha * val + (1 - this.alpha) * this._state[key];
      return this._state[key];
    };
    return {
      cx: lerp('cx', cx),
      cy: lerp('cy', cy),
      scale: lerp('scale', scale),
      angle: this._angleSm.smooth(angle),
    };
  }
  reset() { this._state = {}; this._angleSm.reset(); }
}

// ─── Perspective quad drawing (Canvas 2D triangle-warp) ──────────────────────
/**
 * Draw `img` warped into a 4-point quad using two affine triangles.
 * quad = [[tlX,tlY],[trX,trY],[brX,brY],[blX,blY]]
 * Matches Python perspective_warp() intent.
 */
function drawPerspectiveQuad(ctx, img, quad, alpha = 1.0) {
  const w = img.naturalWidth || img.width || 1;
  const h = img.naturalHeight || img.height || 1;
  // Triangle 1: TL→TR→BR  |  Triangle 2: TL→BR→BL
  _drawTriangle(ctx, img,
    quad[0], quad[1], quad[2],
    { x: 0, y: 0 }, { x: w, y: 0 }, { x: w, y: h }, alpha);
  _drawTriangle(ctx, img,
    quad[0], quad[2], quad[3],
    { x: 0, y: 0 }, { x: w, y: h }, { x: 0, y: h }, alpha);
}

/**
 * Affine-transform img so src triangle (s0,s1,s2) maps to dst triangle (d0,d1,d2).
 * Clips drawing to the destination triangle for clean edges.
 */
function _drawTriangle(ctx, img, d0, d1, d2, s0, s1, s2, alpha) {
  // Compute affine matrix M such that M·s = d for each corresponding point
  // [d0 d1 d2] = M · [s0 s1 s2]  →  M = D · inv(S)
  const det = (s1.x - s0.x) * (s2.y - s0.y) - (s2.x - s0.x) * (s1.y - s0.y);
  if (Math.abs(det) < 1e-6) return;
  const m00 = ((d1.x - d0.x) * (s2.y - s0.y) - (d2.x - d0.x) * (s1.y - s0.y)) / det;
  const m01 = ((d2.x - d0.x) * (s1.x - s0.x) - (d1.x - d0.x) * (s2.x - s0.x)) / det;
  const m10 = ((d1.y - d0.y) * (s2.y - s0.y) - (d2.y - d0.y) * (s1.y - s0.y)) / det;
  const m11 = ((d2.y - d0.y) * (s1.x - s0.x) - (d1.y - d0.y) * (s2.x - s0.x)) / det;
  const dx = d0.x - m00 * s0.x - m01 * s0.y;
  const dy = d0.y - m10 * s0.x - m11 * s0.y;

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.moveTo(d0.x, d0.y);
  ctx.lineTo(d1.x, d1.y);
  ctx.lineTo(d2.x, d2.y);
  ctx.closePath();
  ctx.clip();
  ctx.transform(m00, m10, m01, m11, dx, dy);
  ctx.drawImage(img, 0, 0);
  ctx.restore();
}

/** Draw an image centred at (cx, cy) rotated by angleDeg, sized w×h */
function drawCentredImage(ctx, img, cx, cy, w, h, angleDeg, alpha = 1.0) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.translate(cx, cy);
  ctx.rotate(angleDeg * Math.PI / 180);
  ctx.drawImage(img, -w / 2, -h / 2, w, h);
  ctx.restore();
}

/**
 * Build a 4-point perspective quad centred at (cx,cy).
 * axisUnit = {x,y} unit vector along the finger/wrist long axis.
 * Returns [[tl],[tr],[br],[bl]] screen coords.
 * Ported from Python build_perspective_quad().
 */
function buildPerspectiveQuad(cx, cy, width, height, axisUnit) {
  const { x: ux, y: uy } = axisUnit;
  const nx = -uy, ny = ux;           // perpendicular (across)
  const hw = width / 2, hh = height / 2;
  return [
    { x: cx - nx * hw - ux * hh, y: cy - ny * hw - uy * hh },  // TL
    { x: cx + nx * hw - ux * hh, y: cy + ny * hw - uy * hh },  // TR
    { x: cx + nx * hw + ux * hh, y: cy + ny * hw + uy * hh },  // BR
    { x: cx - nx * hw + ux * hh, y: cy - ny * hw + uy * hh },  // BL
  ];
}

/**
 * Build perspective quad for bracelet using wrist + knuckle anchors.
 * Ported from Python build_wrist_perspective_quad().
 * s_idx, s_pink, s_wrist = display-space pixel {x,y} points
 */
function buildWristPerspectiveQuad(sIdx, sPink, sWrist, bw, bh, depthZ = 0, axisUnit = null) {
  const widthAxis = axisUnit || unitVec2D(sIdx, sPink);
  const knuckleMid = { x: (sIdx.x + sPink.x) / 2, y: (sIdx.y + sPink.y) / 2 };
  const forearmDir = unitVec2D(knuckleMid, sWrist);
  const hw = bw / 2;
  // Depth-based foreshortening (depth_z < 0 → wrist toward camera)
  const foreshortenH = clamp(1.0 + depthZ * 3.0, 0.35, 1.0);
  const hh = (bh / 2) * foreshortenH;
  const palmOff = { x: forearmDir.x * hh, y: forearmDir.y * hh };
  const cx = sWrist.x, cy = sWrist.y;
  return [
    { x: cx - widthAxis.x * hw - palmOff.x, y: cy - widthAxis.y * hw - palmOff.y },
    { x: cx + widthAxis.x * hw - palmOff.x, y: cy + widthAxis.y * hw - palmOff.y },
    { x: cx + widthAxis.x * hw + palmOff.x, y: cy + widthAxis.y * hw + palmOff.y },
    { x: cx - widthAxis.x * hw + palmOff.x, y: cy - widthAxis.y * hw + palmOff.y },
  ];
}

// ─── JewelryAssets ───────────────────────────────────────────────────────────

class JewelryAssets {
  constructor() {
    /**
     * catalog[type] = [ { label, src, img } ]
     */
    this.catalog = {
      ring: [],
      bracelet: [],
      necklace: [],
      earrings: [],
    };
    this._ready = false;
  }

  async load() {
    const defs = [
      { type: 'ring', label: 'Ring 1', src: 'catalog_pack/ring/processed/processed_s1.png' },
      { type: 'ring', label: 'Ring 2', src: 'catalog_pack/ring/processed/processed_s2.png' },
      { type: 'ring', label: 'Ring 3', src: 'catalog_pack/ring/processed/processed_s3.png' },
      { type: 'ring', label: 'Ring 4', src: 'catalog_pack/ring/processed/processed_s4.png' },
      { type: 'bracelet', label: 'Bracelet 1', src: 'catalog_pack/bracelet/processed/processed_s1.png' },
      { type: 'bracelet', label: 'Bracelet 2', src: 'catalog_pack/bracelet/processed/processed_s2.png' },
      { type: 'bracelet', label: 'Bracelet 3', src: 'catalog_pack/bracelet/processed/processed_s3.png' },
      { type: 'necklace', label: 'Necklace 1', src: 'catalog_pack/necklace/necklace_1.png' },
      { type: 'necklace', label: 'Necklace 2', src: 'catalog_pack/necklace/necklace_2.png' },
      { type: 'necklace', label: 'Necklace 3', src: 'catalog_pack/necklace/necklace_3.png' },
      { type: 'necklace', label: 'Necklace 4', src: 'catalog_pack/necklace/necklace_4.png' },
      { type: 'necklace', label: 'Necklace 5', src: 'catalog_pack/necklace/necklace_5.png' },
      { type: 'earrings', label: 'Gold Chandelier', src: 'catalog_pack/earrings/Elaborate gold chandelier earrings in focus.png' },
    ];

    await Promise.all(defs.map(d => this._loadEntry(d)));
    this._ready = true;
  }

  _loadEntry({ type, label, src }) {
    return new Promise(resolve => {
      const img = new Image();
      if (src.startsWith('http')) img.crossOrigin = 'anonymous';
      img.onload = () => { this.catalog[type].push({ label, src, img }); resolve(); };
      img.onerror = () => {
        console.warn(`Failed to load ${src} — generating placeholder`);
        const placeholder = this._generatePlaceholder(type);
        this.catalog[type].push({ label, src, img: placeholder });
        resolve();
      };
      img.src = src;
    });
  }

  /** Fallback canvas-drawn placeholder jewelry */
  _generatePlaceholder(type) {
    const c = document.createElement('canvas');
    c.width = 200; c.height = 200;
    const ctx = c.getContext('2d');

    if (type === 'ring') {
      // Simple drawn ring
      const grd = ctx.createRadialGradient(100, 100, 30, 100, 100, 80);
      grd.addColorStop(0, '#FFD700');
      grd.addColorStop(0.4, '#DAA520');
      grd.addColorStop(1, '#B8860B');
      ctx.beginPath();
      ctx.arc(100, 100, 78, 0, Math.PI * 2);
      ctx.fillStyle = grd;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(100, 100, 50, 0, Math.PI * 2);
      ctx.fillStyle = 'transparent';
      ctx.clearRect(0, 0, 200, 200);
      // Band
      ctx.beginPath();
      ctx.arc(100, 100, 72, 0, Math.PI * 2);
      ctx.strokeStyle = '#FFD700';
      ctx.lineWidth = 14;
      ctx.stroke();
      // Stone
      ctx.beginPath();
      ctx.arc(100, 30, 16, 0, Math.PI * 2);
      const sg = ctx.createRadialGradient(100, 30, 2, 100, 30, 16);
      sg.addColorStop(0, '#ffffff');
      sg.addColorStop(0.5, '#aaeeff');
      sg.addColorStop(1, '#5588cc');
      ctx.fillStyle = sg;
      ctx.fill();

    } else if (type === 'bracelet') {
      // Simple chain
      ctx.strokeStyle = '#FFD700';
      ctx.lineWidth = 10;
      ctx.beginPath();
      ctx.arc(100, 100, 80, 0, Math.PI * 2);
      ctx.stroke();
      for (let i = 0; i < 12; i++) {
        const a = (i / 12) * Math.PI * 2;
        ctx.save();
        ctx.translate(100 + Math.cos(a) * 80, 100 + Math.sin(a) * 80);
        ctx.rotate(a + Math.PI / 2);
        ctx.strokeStyle = '#DAA520';
        ctx.lineWidth = 6;
        ctx.beginPath();
        ctx.ellipse(0, 0, 5, 9, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

    } else if (type === 'necklace') {
      // Simple necklace
      ctx.strokeStyle = '#FFD700';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.arc(100, 60, 80, 0.15 * Math.PI, 0.85 * Math.PI);
      ctx.stroke();
      // Pendant
      ctx.save();
      ctx.translate(100, 140);
      ctx.beginPath();
      ctx.moveTo(0, -20); ctx.lineTo(12, 0); ctx.lineTo(0, 20); ctx.lineTo(-12, 0);
      ctx.closePath();
      const pg = ctx.createLinearGradient(-12, -20, 12, 20);
      pg.addColorStop(0, '#ffffff');
      pg.addColorStop(0.5, '#aaeeff');
      pg.addColorStop(1, '#5588cc');
      ctx.fillStyle = pg;
      ctx.fill();
      ctx.restore();
    }

    const img = new Image();
    img.src = c.toDataURL();
    return img;
  }

  addCustom(type, label, img) {
    this.catalog[type].unshift({ label, src: img.src, img });
  }

  get(type, index = 0) {
    return this.catalog[type]?.[index] ?? null;
  }

  listForType(type) {
    return this.catalog[type] ?? [];
  }
}

// ─── HandTracker ─────────────────────────────────────────────────────────────

class HandTracker {
  constructor() {
    this.results = null;
    this._ready = false;
    this._hands = null;
  }

  async init(onResults) {
    this._hands = new Hands({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/${file}`;
      },
    });

    this._hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.55,
      minTrackingConfidence: 0.55,
    });

    this._hands.onResults(r => {
      this.results = r;
      onResults(r);
    });

    await this._hands.initialize();
    this._ready = true;
    console.log('[HandTracker] ready');
  }

  async send(videoEl) {
    if (!this._ready) return;
    await this._hands.send({ image: videoEl });
  }

  getLandmarks(handIndex = 0) {
    return this.results?.multiHandLandmarks?.[handIndex] ?? null;
  }

  getHandedness(handIndex = 0) {
    return this.results?.multiHandedness?.[handIndex]?.label ?? 'Unknown';
  }

  getHandCount() {
    return this.results?.multiHandLandmarks?.length ?? 0;
  }
}

// ─── FaceTracker ─────────────────────────────────────────────────────────────

class FaceTracker {
  constructor() {
    this.results = null;
    this._ready = false;
    this._mesh = null;
  }

  async init(onResults) {
    this._mesh = new FaceMesh({
      locateFile: (file) => {
        // Unique query param prevents WASM binary namespace clash with Hands
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/${file}`;
      },
    });

    this._mesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    this._mesh.onResults(r => {
      this.results = r;
      onResults(r);
    });

    await this._mesh.initialize();
    this._ready = true;
    console.log('[FaceTracker] ready');
  }

  async send(videoEl) {
    if (!this._ready) return;
    await this._mesh.send({ image: videoEl });
  }

  getLandmarks(faceIndex = 0) {
    return this.results?.multiFaceLandmarks?.[faceIndex] ?? null;
  }

  getFaceCount() {
    return this.results?.multiFaceLandmarks?.length ?? 0;
  }
}

// ─── Renderer ────────────────────────────────────────────────────────────────

class Renderer {
  constructor(canvas, video) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.video = video;

    // High-quality rendering
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = 'high';

    // One JewelrySmoother per jewelry type (ported from Python)
    this._ringSm = new JewelrySmoother(EMA_ALPHA, 0.05);
    this._braceletSm = new JewelrySmoother(EMA_ALPHA, 0.05);
    this._necklaceSm = new JewelrySmoother(EMA_ALPHA, 0.08);
    this._earLSm = new JewelrySmoother(EMA_ALPHA, 0.08);
    this._earRSm = new JewelrySmoother(EMA_ALPHA, 0.08);
  }

  resize(w, h) {
    this.canvas.width = w;
    this.canvas.height = h;
  }

  // ── Per-frame draw ────────────────────────────────────────────
  drawFrame({
    handResults, faceResults,
    assets, activeType, activeIdx,
    scaleMult, alpha, occlusionOn, debugOn, mirrored,
    infoCallback,
  }) {
    const ctx = this.ctx;
    const W = this.canvas.width, H = this.canvas.height;
    if (W === 0 || H === 0) return;

    // 1. Clear
    ctx.clearRect(0, 0, W, H);

    // 2. Draw video (always mirrored to match CSS scaleX(-1)) — high quality
    if (this.video.readyState >= 2 && this.video.videoWidth > 0) {
      ctx.save();
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.scale(-1, 1);
      ctx.drawImage(this.video, -W, 0, W, H);
      ctx.restore();
    }

    const assetEntry = assets.get(activeType, activeIdx);
    const img = assetEntry?.img;
    const infoData = { hands: 0, face: false, fingerPx: null, wristPx: null, jawPx: null };

    // 3. Hand jewelry
    if (img && (activeType === 'ring' || activeType === 'bracelet')) {
      const slms = handResults?.multiHandLandmarks ?? [];
      const wlms = handResults?.multiHandWorldLandmarks ?? [];
      infoData.hands = slms.length;

      for (let hi = 0; hi < slms.length; hi++) {
        const lm = slms[hi];
        const wlm = wlms[hi] ?? null;

        if (activeType === 'ring') {
          this._drawRing(ctx, lm, wlm, img, W, H, scaleMult, alpha, occlusionOn, debugOn);
          // Info: screen-pixel MCP→PIP distance
          const a = px(lm[13], W, H), b = px(lm[14], W, H);
          infoData.fingerPx = Math.round(dist2D(a, b));
        } else {
          this._drawBracelet(ctx, lm, wlm, img, W, H, scaleMult, alpha, occlusionOn, debugOn);
          const a = px(lm[5], W, H), b = px(lm[17], W, H);
          infoData.wristPx = Math.round(dist2D(a, b));
        }
        if (debugOn) this._drawHandDebug(ctx, lm, W, H);
      }
    }

    // 4. Face jewelry (necklace or earrings)
    if (img && (activeType === 'necklace' || activeType === 'earrings')) {
      const faceLm = faceResults?.multiFaceLandmarks?.[0];
      infoData.face = !!faceLm;
      if (faceLm) {
        if (activeType === 'necklace') {
          this._drawNecklace(ctx, faceLm, img, W, H, scaleMult, alpha, debugOn);
          const a = px(faceLm[234], W, H), b = px(faceLm[454], W, H);
          infoData.jawPx = Math.round(dist2D(a, b));
        } else {
          this._drawEarrings(ctx, faceLm, img, W, H, scaleMult, alpha, debugOn);
          const a = px(faceLm[234], W, H), b = px(faceLm[454], W, H);
          infoData.jawPx = Math.round(dist2D(a, b));
        }
      }
    }

    if (infoCallback) infoCallback(infoData);
  }

  // ── RING — ported from Python hand_tryon._apply_ring ──────────────────
  // t=0.45, ring width from world 3D cross-section × pxPerM × depthFac

  _drawRing(ctx, lm, wlm, img, W, H, scaleMult, alpha, occlusionOn, debugOn) {
    const sMCP = px(lm[13], W, H);
    const sPIP = px(lm[14], W, H);

    const segVec = { x: sPIP.x - sMCP.x, y: sPIP.y - sMCP.y };
    const segLen = Math.hypot(segVec.x, segVec.y);
    if (segLen < 2) return;

    let ringW_raw;
    if (wlm) {
      // World landmark cross-section (Python: finger_w_world = cross_dist * w_factor)
      const wMCP = wlm[13], wPIP = wlm[14], wL = wlm[9], wR = wlm[17];
      const worldSegLen = Math.hypot(
        wPIP.x - wMCP.x, wPIP.y - wMCP.y, wPIP.z - wMCP.z) || 1e-9;
      const pxPerM = segLen / worldSegLen;

      // Axis unit in world 2D projection
      const wdLen = Math.hypot(wPIP.x - wMCP.x, wPIP.y - wMCP.y) || 1;
      const wAxis = { x: (wPIP.x - wMCP.x) / wdLen, y: (wPIP.y - wMCP.y) / wdLen };
      const wPerp = { x: -wAxis.y, y: wAxis.x };

      // Project neighboring MCP landmarks onto perpendicular axis
      const projL = (wL.x - wMCP.x) * wPerp.x + (wL.y - wMCP.y) * wPerp.y;
      const projR = (wR.x - wMCP.x) * wPerp.x + (wR.y - wMCP.y) * wPerp.y;
      const fingerWWorld = Math.abs(projL - projR) * 0.55;  // w_factor for Ring

      const depthZ = wMCP.z;
      const depthFac = clamp(1.0 - depthZ * 5.5, 0.60, 1.50);
      ringW_raw = Math.max(fingerWWorld * pxPerM * depthFac * (scaleMult / 100), 8);
    } else {
      // Fallback: proportional to screen segment
      ringW_raw = Math.max(segLen * 0.55 * (scaleMult / 100), 8);
    }

    // t=0.45: ring center is 45% up from MCP toward PIP
    const cx_raw = sMCP.x + 0.45 * segVec.x;
    const cy_raw = sMCP.y + 0.45 * segVec.y;

    const screenDir = unitVec2D(sMCP, sPIP);
    const rawAngle = Math.atan2(screenDir.y, screenDir.x) * 180 / Math.PI;
    const angleDeg = rawAngle + 90; 

    const sm = this._ringSm.smooth({ cx: cx_raw, cy: cy_raw, scale: ringW_raw, angle: angleDeg });
    const rw = Math.max(sm.scale, 8);
    const imgAR = (img.naturalHeight || img.height || 1) / (img.naturalWidth || img.width || 1);
    const rh = rw * imgAR;

    // Use SMOOTHED angle for quad axis! (Fixes shaking)
    const smoothAngleRad = (sm.angle + 180) * Math.PI / 180; // FLIP 180 here
    const smoothAxis = { x: Math.cos(smoothAngleRad - Math.PI / 2), y: Math.sin(smoothAngleRad - Math.PI / 2) };
    const quad = buildPerspectiveQuad(sm.cx, sm.cy, rw, rh, smoothAxis);
    drawPerspectiveQuad(ctx, img, quad, alpha);

    if (occlusionOn) this._ringOcclusion(ctx, sMCP, sPIP, segLen);

    if (debugOn) {
      ctx.fillStyle = 'rgba(255,220,0,0.9)';
      ctx.beginPath(); ctx.arc(sm.cx, sm.cy, 6, 0, 2 * Math.PI); ctx.fill();
      ctx.strokeStyle = 'rgba(255,220,0,0.7)'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(sMCP.x, sMCP.y); ctx.lineTo(sPIP.x, sPIP.y); ctx.stroke();
      ctx.strokeStyle = 'rgba(255,220,0,0.4)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(quad[0].x, quad[0].y);
      quad.forEach(p => ctx.lineTo(p.x, p.y)); ctx.closePath(); ctx.stroke();
    }
  }

  _ringOcclusion(ctx, sMCP, sPIP, segLen) {
    const halfW = segLen * 0.20;
    const dir = unitVec2D(sMCP, sPIP);
    const nx = -dir.y, ny = dir.x;
    const poly = [
      { x: sMCP.x + nx * halfW, y: sMCP.y + ny * halfW },
      { x: sMCP.x - nx * halfW, y: sMCP.y - ny * halfW },
      { x: sPIP.x - nx * halfW, y: sPIP.y - ny * halfW },
      { x: sPIP.x + nx * halfW, y: sPIP.y + ny * halfW },
    ];
    ctx.save();
    ctx.beginPath(); ctx.moveTo(poly[0].x, poly[0].y);
    poly.forEach(p => ctx.lineTo(p.x, p.y)); ctx.closePath();
    const g = ctx.createLinearGradient(
      sMCP.x - nx * halfW, sMCP.y - ny * halfW, sMCP.x + nx * halfW, sMCP.y + ny * halfW);
    g.addColorStop(0, 'rgba(210,160,120,0.0)');
    g.addColorStop(0.3, 'rgba(215,168,128,0.40)');
    g.addColorStop(0.5, 'rgba(222,175,135,0.60)');
    g.addColorStop(0.7, 'rgba(215,168,128,0.40)');
    g.addColorStop(1, 'rgba(210,160,120,0.0)');
    ctx.fillStyle = g; ctx.fill(); ctx.restore();
  }

  // ── BRACELET — ported from Python hand_tryon._apply_bracelet ───────────
  // knuckle_span*0.75*depthFac, center=wrist shifted 30% toward forearm

  _drawBracelet(ctx, lm, wlm, img, W, H, scaleMult, alpha, occlusionOn, debugOn) {
    const sWrist = px(lm[0], W, H);
    const sIdx = px(lm[5], W, H);
    const sPink = px(lm[17], W, H);

    const knuckleSpan = dist2D(sIdx, sPink);
    if (knuckleSpan < 5) return;

    const depthZ = wlm ? wlm[0].z : 0;
    const depthFac = clamp(1.0 - depthZ * 4.5, 0.60, 1.50);

    // Python: bracelet_w_raw = knuckle_span * 0.75 * depth_fac * scale_mul
    const bw_raw = Math.max(knuckleSpan * 0.75 * depthFac * (scaleMult / 100), 12);

    // Angle: index → pinky direction + slight tilt to align with wrist instead of knuckles
    const wristVec = unitVec2D(sIdx, sPink);
    const angleDeg = Math.atan2(wristVec.y, wristVec.x) * 180 / Math.PI + 10;

    // Python: shift wrist 30% further toward forearm from knuckle midpoint
    const knuckleMid = { x: (sIdx.x + sPink.x) / 2, y: (sIdx.y + sPink.y) / 2 };
    const fwVec = { x: sWrist.x - knuckleMid.x, y: sWrist.y - knuckleMid.y };
    const fwLen = Math.hypot(fwVec.x, fwVec.y) || 1;
    const fwUnit = { x: fwVec.x / fwLen, y: fwVec.y / fwLen };
    const sWristLow = {
      x: sWrist.x + fwUnit.x * fwLen * 0.30,
      y: sWrist.y + fwUnit.y * fwLen * 0.30,
    };

    const sm = this._braceletSm.smooth({ cx: sWristLow.x, cy: sWristLow.y, scale: bw_raw, angle: angleDeg });
    const bw = Math.max(sm.scale, 12);
    const imgAR = (img.naturalHeight || img.height || 1) / (img.naturalWidth || img.width || 1);
    const bh = bw * imgAR;

    // Use SMOOTHED angle for quad axis! (Fixes shaking)
    const smoothAngleRad = sm.angle * Math.PI / 180;
    const smoothAxis = { x: Math.cos(smoothAngleRad), y: Math.sin(smoothAngleRad) };
    const quad = buildWristPerspectiveQuad(sIdx, sPink, { x: sm.cx, y: sm.cy }, bw, bh, depthZ, smoothAxis);
    drawPerspectiveQuad(ctx, img, quad, alpha);

    if (debugOn) {
      ctx.fillStyle = 'rgba(0,220,255,0.9)';
      [sWrist, sIdx, sPink, { x: sm.cx, y: sm.cy }].forEach(p => {
        ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, 2 * Math.PI); ctx.fill();
      });
      ctx.strokeStyle = 'rgba(0,220,255,0.7)'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(sIdx.x, sIdx.y); ctx.lineTo(sPink.x, sPink.y); ctx.stroke();
      ctx.strokeStyle = 'rgba(0,220,255,0.4)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(quad[0].x, quad[0].y);
      quad.forEach(p => ctx.lineTo(p.x, p.y)); ctx.closePath(); ctx.stroke();
    }
  }

  // ── NECKLACE — ported from Python face_tryon._apply_necklace ─────────
  // cy = chin_y + face_h*1.10, scale = jaw_dist*1.20*z_factor

  _drawNecklace(ctx, faceLm, img, W, H, scaleMult, alpha, debugOn) {
    const pLJaw = px(faceLm[234], W, H);  // person's left → visually RIGHT in mirror
    const pRJaw = px(faceLm[454], W, H);  // person's right → visually LEFT in mirror
    const pChin = px(faceLm[152], W, H);
    const pNose = px(faceLm[1], W, H);

    const jawDist = dist2D(pLJaw, pRJaw);
    const faceH = Math.abs(pChin.y - pNose.y);
    if (jawDist < 10 || faceH < 10) return;

    const cx_raw = (pLJaw.x + pRJaw.x) / 2;
    const cy_raw = pChin.y + faceH * 0.90;

    const zAvg = (faceLm[234].z + faceLm[454].z) / 2;
    const zFactor = clamp(1.0 - zAvg * 1.5, 0.7, 1.3);
    const scale_raw = Math.max(jawDist * 1.20 * zFactor * (scaleMult / 100), 10);

    // Fix: In mirrored display, LM234 is visually on the RIGHT and LM454 on the LEFT.
    // The jaw vector in screen space goes from pRJaw (left) → pLJaw (right).
    // Python uses -calculate_angle(ljaw, rjaw) on the unmirrored frame where
    // ljaw.x < rjaw.x, giving a small angle. In mirrored coords pLJaw.x > pRJaw.x,
    // so we compute atan2 from pRJaw to pLJaw to get the same small left-to-right angle.
    const angleDeg = Math.atan2(pLJaw.y - pRJaw.y, pLJaw.x - pRJaw.x) * 180 / Math.PI;

    const sm = this._necklaceSm.smooth({ cx: cx_raw, cy: cy_raw, scale: scale_raw, angle: angleDeg });
    const nw = Math.max(sm.scale, 10);
    const imgAR = (img.naturalHeight || img.height || 1) / (img.naturalWidth || img.width || 1);
    const nh = nw * imgAR;

    drawCentredImage(ctx, img, sm.cx, sm.cy, nw, nh, sm.angle, alpha);

    if (debugOn) {
      ctx.fillStyle = 'rgba(255,80,200,0.9)';
      [pLJaw, pRJaw, pChin, pNose, { x: sm.cx, y: sm.cy }].forEach(p => {
        ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, 2 * Math.PI); ctx.fill();
      });
      ctx.strokeStyle = 'rgba(255,80,200,0.7)'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(pLJaw.x, pLJaw.y); ctx.lineTo(pRJaw.x, pRJaw.y); ctx.stroke();
    }
  }

  // ── EARRINGS — ported from Python face_tryon._apply_earrings ────────────
  // L_LOBE=177, R_LOBE=401, L_EAR_UP=127, R_EAR_UP=356
  // ear_sz = face_w*0.22, push outward by face_w*0.05

  _drawEarrings(ctx, faceLm, img, W, H, scaleMult, alpha, debugOn) {
    const pLJaw = px(faceLm[234], W, H);
    const pRJaw = px(faceLm[454], W, H);
    const pNose = px(faceLm[1], W, H);

    // Earlobe anchors (pulling them far out to the extreme edges to prevent face overlap)
    const pLLobe = px(faceLm[234], W, H);  
    const pRLobe = px(faceLm[454], W, H);  
    const pLUp = px(faceLm[127], W, H);  // person's left ear top → visually RIGHT
    const pRUp = px(faceLm[356], W, H);  // person's right ear top → visually LEFT

    const faceW = dist2D(pLJaw, pRJaw);
    const earSz = Math.max(faceW * 0.22 * (scaleMult / 100), 16);
    const imgAR = (img.naturalHeight || img.height || 1) / (img.naturalWidth || img.width || 1);
    const earDrop = earSz * imgAR;          // how far the pendant hangs
    const pushPx = faceW * 0.12;           // push earring outward from lobe to clear the face
    const noseCx = pNose.x;

    // —— Left earring (person's left → visually RIGHT of screen) ——
    // In mirrored display, pLLobe.x > noseCx, so push rightward (outward)
    const lPush = pLLobe.x >= noseCx ? pushPx : -pushPx;
    const lCx_raw = pLLobe.x + lPush;
    const lCy_raw = pLLobe.y + earDrop * 0.8; // Lowered to position on lobe
    // Angle: direction ear-top → lobe in screen space, then perpendicular
    const lEarAngle = Math.atan2(pLLobe.y - pLUp.y, pLLobe.x - pLUp.x) * 180 / Math.PI;
    const lAngle_raw = -lEarAngle + 90;

    const smL = this._earLSm.smooth({ cx: lCx_raw, cy: lCy_raw, scale: earSz, angle: lAngle_raw });
    drawCentredImage(ctx, img, smL.cx, smL.cy, Math.max(smL.scale, 8), Math.max(smL.scale, 8) * imgAR, smL.angle, alpha);

    // —— Right earring (person's right → visually LEFT of screen) ——
    // In mirrored display, pRLobe.x < noseCx, so push leftward (outward)
    const rPush = pRLobe.x < noseCx ? -pushPx : pushPx;
    const rCx_raw = pRLobe.x + rPush;
    const rCy_raw = pRLobe.y + earDrop * 0.8; // Lowered to position on lobe
    const rEarAngle = Math.atan2(pRLobe.y - pRUp.y, pRLobe.x - pRUp.x) * 180 / Math.PI;
    const rAngle_raw = -rEarAngle + 90;

    const smR = this._earRSm.smooth({ cx: rCx_raw, cy: rCy_raw, scale: earSz, angle: rAngle_raw });
    drawCentredImage(ctx, img, smR.cx, smR.cy, Math.max(smR.scale, 8), Math.max(smR.scale, 8) * imgAR, smR.angle, alpha);

    if (debugOn) {
      ctx.fillStyle = 'rgba(255,150,0,0.9)';
      [pLLobe, pRLobe, pLUp, pRUp].forEach(p => {
        ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, 2 * Math.PI); ctx.fill();
      });
    }
  }

  // ── Hand Debug ─────────────────────────────────────────────

  _drawHandDebug(ctx, lm, W, H) {
    const kpts = [0, 5, 9, 13, 14, 15, 17];
    ctx.fillStyle = 'rgba(255,220,0,0.9)';
    kpts.forEach(i => {
      if (!lm[i]) return;
      const p = px(lm[i], W, H);
      ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, 2 * Math.PI); ctx.fill();
    });
    const w = px(lm[0], W, H);
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    ctx.font = '11px Inter,sans-serif';
    ctx.fillText('W', w.x + 6, w.y - 4);
  }
}
// ─── VTOApp  (main controller) ───────────────────────────────────────────────

class VTOApp {
  constructor() {
    // DOM refs
    this.videoEl = document.getElementById('webcam-video');
    this.canvas = document.getElementById('output-canvas');
    this.thumbCanvas = document.getElementById('thumb-canvas');

    this.loadingEl = document.getElementById('cam-loading');
    this.nopermEl = document.getElementById('cam-noperm');
    this.statusDot = document.getElementById('status-dot');
    this.statusTxt = document.getElementById('status-text');
    this.fpsCtr = document.getElementById('fps-counter');

    // State
    this.activeType = 'ring';
    this.activeIdx = 0;
    this.scaleMult = 100;   // from slider
    this.alphaPct = 95;    // from opacity slider
    this.occlusionOn = true;
    this.debugOn = false;
    this.mirrored = true;
    this.running = false;
    this.frameCount = 0;
    this.lastFpsTs = performance.now();

    // Sub-systems
    this.assets = new JewelryAssets();
    this.handTracker = new HandTracker();
    this.faceTracker = new FaceTracker();
    this.renderer = new Renderer(this.canvas, this.videoEl);

    // Initialize canvas size to wrapper dimensions.
    // Use setTimeout to defer until after first layout paint (getBoundingClientRect
    // returns 0x0 if called synchronously before layout)
    const wrapper = document.getElementById('camera-wrapper');
    const doInitialResize = () => {
      if (wrapper) {
        const rect = wrapper.getBoundingClientRect();
        const initW = Math.round(rect.width) || CAM_WIDTH;
        const initH = Math.round(rect.height) || CAM_HEIGHT;
        if (initW > 0 && initH > 0) {
          this.renderer.resize(initW, initH);
          console.log(`[Canvas] initialized to ${initW}x${initH}`);
        }
      }
    };
    // Try immediately, then again after a tick (layout may not have run yet)
    doInitialResize();
    setTimeout(doInitialResize, 100);

    // Keep canvas sized to wrapper on window resize (when no real video yet)
    if (window.ResizeObserver && wrapper) {
      new ResizeObserver(entries => {
        const { width, height } = entries[0].contentRect;
        if (this.videoEl.videoWidth === 0 && width > 0 && height > 0) {
          this.renderer.resize(Math.round(width), Math.round(height));
        }
      }).observe(wrapper);
    }

    // Latest tracking results
    this._handRes = null;
    this._faceRes = null;

    // Animation
    this._rafId = null;
  }

  // ── Initialisation ────────────────────────────────────

  async start() {
    this._updateStatus('Loading models…', 'warn');

    // Load assets first
    await this.assets.load();

    // Init Hands tracker immediately (used for Ring and Bracelet)
    try {
      await this.handTracker.init(r => { this._handRes = r; });
      console.log('[VTOApp] HandTracker initialized');
    } catch (e) {
      console.error('[HandTracker init]', e);
      this._updateStatus('Hand model failed', 'warn');
    }

    // FaceTracker is lazily initialized on first necklace selection
    // (avoids WASM binary namespace conflict when both load simultaneously)

    this._buildUI();
    await this._startCamera();
  }

  /** Lazy-initialize FaceTracker on first necklace use */
  async _ensureFaceTracker() {
    if (this.faceTracker._ready) return true;
    this._updateStatus('Loading face model…', 'warn');
    try {
      // 500ms delay ensures Hands WASM has fully settled its global state
      await new Promise(r => setTimeout(r, 500));
      await this.faceTracker.init(r => { this._faceRes = r; });
      this._updateStatus('Tracking active', 'ok');
      console.log('[VTOApp] FaceTracker initialized');
      return true;
    } catch (e) {
      console.error('[FaceTracker lazy init]', e);
      this._updateStatus('Face model failed', 'error');
      return false;
    }
  }

  // ── Camera ────────────────────────────────────────────

  async _startCamera() {
    this._updateStatus('Requesting camera…', 'warn');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: CAM_WIDTH },
          height: { ideal: CAM_HEIGHT },
          facingMode: 'user',
          frameRate: { ideal: 60 },
        },
        audio: false,
      });

      this.videoEl.srcObject = stream;

      // Wait for `canplay` – this is when videoWidth/Height are guaranteed non-zero
      await new Promise(res => {
        this.videoEl.oncanplay = () => res();
        this.videoEl.onloadedmetadata = () => this.videoEl.play();
      });

      // Now dimensions are real
      const vW = this.videoEl.videoWidth || CAM_WIDTH;
      const vH = this.videoEl.videoHeight || CAM_HEIGHT;
      this.renderer.resize(vW, vH);
      console.log(`[Camera] stream ready: ${vW}x${vH}`);

      this.loadingEl.classList.add('hidden');
      this._updateStatus('Tracking active', 'ok');
      this.running = true;

      this._loop();

    } catch (err) {
      console.error('[Camera]', err);
      this.loadingEl.classList.add('hidden');
      this.nopermEl.classList.remove('hidden');
      this._updateStatus('Camera denied', 'error');
    }
  }

  // ── Main loop ─────────────────────────────────────────

  _loop() {
    if (!this.running) return;

    const now = performance.now();

    // Auto-resize canvas if video dimensions change (e.g. first frame arrives late)
    const vW = this.videoEl.videoWidth;
    const vH = this.videoEl.videoHeight;
    if (vW > 0 && vH > 0 &&
      (this.canvas.width !== vW || this.canvas.height !== vH)) {
      this.renderer.resize(vW, vH);
    }

    this.renderer.drawFrame({
      handResults: this._handRes,
      faceResults: this._faceRes,
      assets: this.assets,
      activeType: this.activeType,
      activeIdx: this.activeIdx,
      scaleMult: this.scaleMult,
      alpha: this.alphaPct / 100,
      occlusionOn: this.occlusionOn,
      debugOn: this.debugOn,
      mirrored: this.mirrored,
      infoCallback: info => this._updateInfo(info),
    });

    // Feed latest frame into MediaPipe (non-blocking)
    this._sendToTrackers();

    // FPS
    this.frameCount++;
    const elapsed = now - this.lastFpsTs;
    if (elapsed >= 800) {
      const fps = Math.round((this.frameCount / elapsed) * 1000);
      this.fpsCtr.textContent = `${fps} FPS`;
      this.frameCount = 0;
      this.lastFpsTs = now;
    }

    this._rafId = requestAnimationFrame(() => this._loop());
  }

  _sendToTrackers() {
    if (this.videoEl.readyState >= 2) {
      if (this.activeType === 'ring' || this.activeType === 'bracelet') {
        this.handTracker.send(this.videoEl).catch(() => { });
      }
      if (this.activeType === 'necklace' || this.activeType === 'earrings') {
        this.faceTracker.send(this.videoEl).catch(() => { });
      }
    }
  }

  // ── UI construction ───────────────────────────────────

  _buildUI() {
    // Jewelry type toggles
    document.getElementById('jewelry-toggles').addEventListener('click', async e => {
      const btn = e.target.closest('[data-type]');
      if (!btn) return;
      document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      this.activeType = btn.dataset.type;
      this.activeIdx = 0;
      this._buildStyleSwatches();

      // Lazy-load FaceTracker when necklace/earrings is first selected
      if (this.activeType === 'necklace' || this.activeType === 'earrings') {
        await this._ensureFaceTracker();
      }
    });

    // Settings sliders
    const scaleSlider = document.getElementById('scale-slider');
    const opacitySlider = document.getElementById('opacity-slider');
    const scaleVal = document.getElementById('scale-val');
    const opacityVal = document.getElementById('opacity-val');

    const _updateSlider = (slider) => {
      const pct = ((slider.value - slider.min) / (slider.max - slider.min)) * 100;
      slider.style.setProperty('--val', `${pct}%`);
    };

    scaleSlider.addEventListener('input', () => {
      this.scaleMult = +scaleSlider.value;
      scaleVal.textContent = `${this.scaleMult}%`;
      _updateSlider(scaleSlider);
    });
    opacitySlider.addEventListener('input', () => {
      this.alphaPct = +opacitySlider.value;
      opacityVal.textContent = `${this.alphaPct}%`;
      _updateSlider(opacitySlider);
    });
    _updateSlider(scaleSlider);
    _updateSlider(opacitySlider);

    // Toggle switches
    document.getElementById('occlusion-toggle').addEventListener('change', e => {
      this.occlusionOn = e.target.checked;
    });
    document.getElementById('debug-toggle').addEventListener('change', e => {
      this.debugOn = e.target.checked;
    });

    // Action buttons
    document.getElementById('snapshot-btn').addEventListener('click', () => this._takeSnapshot());
    document.getElementById('mirror-btn').addEventListener('click', () => {
      this.mirrored = !this.mirrored;
    });
    document.getElementById('stop-btn').addEventListener('click', () => this._stopCamera());
    document.getElementById('retry-btn').addEventListener('click', () => {
      this.nopermEl.classList.add('hidden');
      this.loadingEl.classList.remove('hidden');
      this._startCamera();
    });

    // Modal
    document.getElementById('modal-close').addEventListener('click', () => this._closeModal());
    document.getElementById('modal-retry').addEventListener('click', () => this._closeModal());

    // Upload
    document.getElementById('upload-input').addEventListener('change', e => {
      const file = e.target.files[0];
      if (!file) return;
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        this.assets.addCustom(this.activeType, file.name.replace(/\.[^.]+$/, ''), img);
        this.activeIdx = 0;
        this._buildStyleSwatches();
      };
      img.src = url;
    });

    // Build initial style swatches for default type
    this._buildStyleSwatches();
  }

  _buildStyleSwatches() {
    const grid = document.getElementById('style-grid');
    grid.innerHTML = '';
    const items = this.assets.listForType(this.activeType);
    items.forEach((item, i) => {
      const swatch = document.createElement('div');
      swatch.className = 'style-swatch' + (i === this.activeIdx ? ' active' : '');
      swatch.title = item.label;

      const img = document.createElement('img');
      img.src = item.src;
      img.alt = item.label;
      img.onerror = () => { img.src = ''; }; // graceful

      const label = document.createElement('div');
      label.className = 'swatch-label';
      label.textContent = item.label.substring(0, 8);

      swatch.appendChild(img);
      swatch.appendChild(label);

      swatch.addEventListener('click', () => {
        this.activeIdx = i;
        grid.querySelectorAll('.style-swatch').forEach(s => s.classList.remove('active'));
        swatch.classList.add('active');
      });

      grid.appendChild(swatch);
    });

    // Placeholder if empty
    if (!items.length) {
      grid.innerHTML = '<p style="color:var(--text-muted);font-size:11px;grid-column:1/-1">No styles loaded yet</p>';
    }
  }

  // ── Info panel ────────────────────────────────────────

  _updateInfo(info) {
    document.getElementById('info-hands').textContent = info.hands > 0 ? `${info.hands} detected` : '–';
    document.getElementById('info-face').textContent = info.face ? 'detected' : '–';
    document.getElementById('info-finger').textContent = info.fingerPx != null ? `${info.fingerPx}px` : '–';
    document.getElementById('info-wrist').textContent = info.wristPx != null ? `${info.wristPx}px` : '–';
    document.getElementById('info-jaw').textContent = info.jawPx != null ? `${info.jawPx}px` : '–';
  }

  // ── Snapshot ──────────────────────────────────────────

  _takeSnapshot() {
    const dataUrl = this.canvas.toDataURL('image/png');

    // Show modal
    const modal = document.getElementById('snapshot-modal');
    document.getElementById('modal-img').src = dataUrl;
    document.getElementById('modal-download').href = dataUrl;
    modal.classList.remove('hidden');

    // Update thumbnail
    const thumb = this.thumbCanvas;
    const tctx = thumb.getContext('2d');
    tctx.clearRect(0, 0, thumb.width, thumb.height);
    tctx.drawImage(this.canvas, 0, 0, thumb.width, thumb.height);
    document.querySelector('.preview-hint').style.display = 'none';
    document.getElementById('download-btn').classList.remove('hidden');

    document.getElementById('download-btn').onclick = () => {
      const a = document.createElement('a');
      a.href = dataUrl;
      a.download = 'luxar-snapshot.png';
      a.click();
    };
  }

  _closeModal() {
    document.getElementById('snapshot-modal').classList.add('hidden');
  }

  // ── Camera stop ───────────────────────────────────────

  _stopCamera() {
    this.running = false;
    if (this._rafId) cancelAnimationFrame(this._rafId);
    const stream = this.videoEl.srcObject;
    if (stream) stream.getTracks().forEach(t => t.stop());
    this.videoEl.srcObject = null;
    this._updateStatus('Camera stopped', 'error');
    this.fpsCtr.textContent = '– FPS';

    // Show a "restart" UI
    this.nopermEl.querySelector('p').textContent = 'Camera stopped.';
    this.nopermEl.querySelector('.retry-btn').textContent = 'Restart';
    this.nopermEl.classList.remove('hidden');
  }

  // ── Status ────────────────────────────────────────────

  _updateStatus(msg, state = '') {
    this.statusTxt.textContent = msg;
    this.statusDot.className = 'status-dot' + (state ? ` ${state}` : '');
  }
}

// ─── Boot ─────────────────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
  const app = new VTOApp();
  app.start().catch(err => {
    console.error('[VTOApp]', err);
  });
});

