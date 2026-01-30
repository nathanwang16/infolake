import time
from typing import Optional
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from src.logging.logger import get_logger

logger = get_logger("crawler")

def fetch_with_playwright(url: str, timeout: int = 30000) -> Optional[str]:
    """
    Fetches a URL using a headless browser (Playwright).
    Returns the page HTML content or None on failure.
    """
    try:
        with sync_playwright() as p:
            # Launch browser
            # headless=True is default, but explicit is good.
            browser = p.chromium.launch(headless=True)
            
            # Create a new context with a realistic user agent to avoid detection
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 720},
                device_scale_factor=1,
            )
            
            page = context.new_page()
            
            # Navigate
            try:
                # wait_until='domcontentloaded' is faster than 'load' (all resources)
                response = page.goto(url, timeout=timeout, wait_until="domcontentloaded")
                
                if not response:
                    logger.warning(f"No response from {url}")
                    browser.close()
                    return None
                
                if response.status >= 400:
                    logger.warning(f"HTTP {response.status} fetching {url}")
                    # Some 403s are bot blocks that might actually display content, but usually not.
                    # We'll return content anyway just in case, but log it.
                
                # Get content
                content = page.content()
                browser.close()
                return content
                
            except PlaywrightTimeoutError:
                logger.warning(f"Timeout fetching {url}")
                browser.close()
                return None
            except Exception as e:
                logger.error(f"Page error fetching {url}: {e}")
                browser.close()
                return None
                
    except Exception as e:
        logger.error(f"Playwright error for {url}: {e}")
        return None
