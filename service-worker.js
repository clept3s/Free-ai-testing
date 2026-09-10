const CACHE_NAME = 'learnai-static-v2';

self.addEventListener('install', event => {
    self.skipWaiting();
});

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

self.addEventListener('fetch', event => {
    const url = new URL(event.request.url);

    // NEVER cache/intercept API calls (POST /generate)
    if (event.request.method !== 'GET' || url.pathname.startsWith('/generate')) {
        return;
    }

    // For HTML/navigation: NETWORK-FIRST (always get fresh page, cache as fallback)
    if (event.request.mode === 'navigate' || url.pathname === '/' || url.pathname.endsWith('.html')) {
        event.respondWith(
            fetch(event.request)
                .then(response => {
                    const copy = response.clone();
                    caches.open(CACHE_NAME).then(cache => cache.put(event.request, copy));
                    return response;
                })
                .catch(() => caches.match(event.request))
        );
        return;
    }

    // For other static assets (icons etc): CACHE-FIRST
    event.respondWith(
        caches.match(event.request).then(cached =>
            cached ||
            fetch(event.request).then(response => {
                const copy = response.clone();
                caches.open(CACHE_NAME).then(cache => cache.put(event.request, copy));
                return response;
            })
        )
    );
});