# Crawler Improvement Proposals

The current crawler uses `trafilatura`'s default fetcher (based on `urllib3`/`requests`), which is fast but easily blocked by modern anti-bot protections (Cloudflare, Akamai) and fails on JavaScript-heavy sites (SPA).

Here are 3 proposed levels of improvement:

## Option 1: Browser Masquerading (Recommended First Step)
Use a library like `curl_cffi` or `tls_client` to mimic real browser TLS fingerprints (JA3 signatures).
*   **Pros:** Fast (C-based), bypasses many static bot filters (Cloudflare), low resource usage.
*   **Cons:** Cannot execute JavaScript (won't fix empty pages that require JS).
*   **Implementation:** Replace `trafilatura.fetch_url` with a custom fetcher using `curl_cffi`.

## Option 2: Headless Browser (The "Powerful" Solution)
Use `Playwright` or `Puppeteer` to run a real headless Chrome/Firefox instance.
*   **Pros:** Highest success rate. Executes JavaScript, handles client-side rendering, manages cookies/sessions automatically.
*   **Cons:** Slow (10x slower than requests), high memory/CPU usage, requires installing browser binaries.
*   **Implementation:** Add `playwright` dependency, install browsers, run fetch in a controlled browser context.

## Option 3: Hybrid Tiered Strategy (Best of Both)
Implement a fetcher that attempts methods in order of cost/complexity:
1.  **Fast Tier:** Standard Request (with timeout).
2.  **Masquerade Tier:** `curl_cffi` with browser headers (if Fast fails).
3.  **Browser Tier:** Playwright (if Masquerade fails or specific errors occur).

*   **Pros:** Balances speed and reliability.
*   **Cons:** Most complex to implement.

## Recommendation
Given the goal "more powerful crawler", **Option 3 (Hybrid)** is the robust engineering choice. However, starting with **Option 2 (Playwright)** guarantees the best immediate results for "access errors" if resources allow.

We can proceed by:
1.  Adding `playwright` and `curl_cffi` to `environment.yml`.
2.  Creating a `src.crawling.fetcher` module.
3.  Updating the calibration script to use this new fetcher.
