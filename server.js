/**
 * server.js
 * ---------
 * Minimal Express server for LuxAR VTO on Railway.
 *
 * WHY THIS EXISTS:
 * MediaPipe WASM uses SharedArrayBuffer which requires these headers
 * or browsers block WASM threads and models fail.
 *
 * Required headers:
 *   Cross-Origin-Opener-Policy: same-origin
 *   Cross-Origin-Embedder-Policy: require-corp
 *
 * Additional header added for CDN compatibility:
 *   Cross-Origin-Resource-Policy: cross-origin
 */

const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

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
app.use(express.static(path.join(__dirname, '.'), {
    maxAge: '1d',
    index: 'index.html',
}));

// ── SPA fallback – always return index.html ───────────────────────────────────
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// ── Start server ──────────────────────────────────────────────────────────────
app.listen(PORT, () => {
    console.log(`[LuxAR] Server running on port ${PORT}`);
    console.log(`[LuxAR] COOP / COEP / CORP headers active`);
    console.log(`[LuxAR] MediaPipe WASM + SharedArrayBuffer enabled`);
});
