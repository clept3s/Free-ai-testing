const CACHE_NAME = "learnai-v1";

const FILES_TO_CACHE = [
    "./",
    "./index.html",
    "./manifest.json"
];


self.addEventListener(
    "install",
    event => {

        event.waitUntil(

            caches.open(CACHE_NAME)
                .then(cache =>
                    cache.addAll(
                        FILES_TO_CACHE
                    )
                )

        );

    }
);


self.addEventListener(
    "activate",
    event => {

        event.waitUntil(

            caches.keys()
                .then(names =>

                    Promise.all(

                        names
                            .filter(
                                name =>
                                    name !== CACHE_NAME
                            )
                            .map(
                                name =>
                                    caches.delete(name)
                            )

                    )

                )

        );

    }
);


self.addEventListener(
    "fetch",
    event => {

        /*
         * Do NOT cache API requests.
         *
         * AI requests need to reach
         * the actual backend.
         */

        if (
            event.request.url.includes("/generate")
        ) {
            return;
        }


        event.respondWith(

            caches.match(event.request)
                .then(cached =>

                    cached ||
                    fetch(event.request)

                )

        );

    }
);
