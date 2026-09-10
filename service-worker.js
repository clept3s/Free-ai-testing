const CACHE_NAME = 'learnai-static-v1';
const ASSETS_TO_CACHE = [
    '/',
    '/index.html',
    '/manifest.json',
    '/icon-192.png',
    '/icon-512.png',
    '/service-worker.js'
];

// Install – cache static assets
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => cache.addAll(ASSETS_TO_CACHE))
    );
    self.skipWaiting();
});

// Activate – clean old caches
self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(keys =>
            Promise.all(
                keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
            )
        )
    );
    self.clients.claim();
});

// Fetch – cache‑first for static assets, network‑only for API calls
self.addEventListener('fetch', event => {
    const url = new URL(event.request.url);
    // Do NOT cache POST /generate or any API request
    if (event.request.method !== 'GET' || url.pathname.startsWith('/generate')) {
        return; // let the request go to network
    }

    // Cache‑first strategy for static assets
    event.respondWith(
        caches.match(event.request).then(cached => {
            return cached || fetch(event.request);
        })
    );
});
