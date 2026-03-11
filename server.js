/**
 * server.js
 * ---------
 * Minimal Express server for LuxAR VTO on Railway.
 *
 * MediaPipe WASM uses SharedArrayBuffer which requires:
 *   Cross-Origin-Opener-Policy: same-origin
 *   Cross-Origin-Embedder-Policy: require-corp
 *
 * Additional header:
 *   Cross-Origin-Resource-Policy: cross-origin
 */

const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 8080;

// Resolve project root
const ROOT_DIR = path.resolve(__dirname);

// ── Security headers required for MediaPipe WASM / SharedArrayBuffer ──────────
app.use((req, res, next) => {

    // Required for SharedArrayBuffer
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');

    // Allows external CDN resources (MediaPipe JS + WASM)
    res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin');

    // Cache static assets
    if (req.path.match(/\.(png|jpg|jpeg|webp|gif|svg|wasm|js|css)$/)) {
        res.setHeader('Cache-Control', 'public, max-age=86400');
    }

    next();
});

// ── Serve static files from project root ──────────────────────────────────────
app.use(express.static(ROOT_DIR, {
    maxAge: '1d'
}));

// ── Healthcheck endpoint (Railway / monitoring) ───────────────────────────────
app.get('/health', (req, res) => {
    res.status(200).send('OK');
});

// ── Root route ────────────────────────────────────────────────────────────────
app.get('/', (req, res) => {
    res.sendFile(path.join(ROOT_DIR, 'index.html'));
});

// ── SPA fallback – return index.html for other routes ─────────────────────────
app.get('*', (req, res) => {
    res.sendFile(path.join(ROOT_DIR, 'index.html'));
});

// ── Basic error handler ───────────────────────────────────────────────────────
app.use((err, req, res, next) => {
    console.error('[LuxAR ERROR]', err);
    res.status(500).send('Server Error');
});

// ── Start server ──────────────────────────────────────────────────────────────
app.listen(PORT, () => {
    console.log(`[LuxAR] Server running on port ${PORT}`);
    console.log(`[LuxAR] COOP / COEP / CORP headers active`);
    console.log(`[LuxAR] MediaPipe WASM + SharedArrayBuffer enabled`);
});
