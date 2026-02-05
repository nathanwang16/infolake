"""
Text processing utilities for the Atlas pipeline.

Provides:
- Content excerpt extraction (first paragraph or first N words)
- Text cleaning and normalization
"""

import re
from typing import Optional

from src.logging.logger import get_logger

logger = get_logger("text_utils")

# Default excerpt length in words
DEFAULT_EXCERPT_WORDS = 100

# Minimum excerpt length to be useful
MIN_EXCERPT_LENGTH = 50


def extract_excerpt(
    text: str,
    max_words: int = DEFAULT_EXCERPT_WORDS,
    prefer_first_paragraph: bool = True
) -> Optional[str]:
    """
    Extracts a content excerpt (first paragraph or first N words).
    
    Strategy:
    1. If prefer_first_paragraph=True, try to get the first meaningful paragraph
    2. Fall back to first max_words words
    3. Clean up and normalize whitespace
    
    Args:
        text: Full document text
        max_words: Maximum number of words for the excerpt
        prefer_first_paragraph: Whether to try extracting first paragraph first
        
    Returns:
        Excerpt string or None if text is too short
    """
    if not text or len(text.strip()) < MIN_EXCERPT_LENGTH:
        return None
    
    # Clean and normalize
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    excerpt = None
    
    if prefer_first_paragraph:
        # Try to find first meaningful paragraph
        # Split on double newlines or paragraph markers
        paragraphs = re.split(r'\n\n+|\r\n\r\n+', text)
        
        for para in paragraphs:
            para = para.strip()
            # Skip very short paragraphs (likely headers or navigation)
            if len(para) >= MIN_EXCERPT_LENGTH:
                # Check it's not a header-like line (all caps, ends with colon, etc.)
                if not _is_header_like(para):
                    excerpt = para
                    break
    
    # If no good paragraph found, fall back to first N words
    if not excerpt:
        words = text.split()
        if len(words) <= max_words:
            excerpt = text
        else:
            excerpt = ' '.join(words[:max_words])
    
    # Truncate to max_words if paragraph is too long
    words = excerpt.split()
    if len(words) > max_words:
        excerpt = ' '.join(words[:max_words])
    
    # Add ellipsis if truncated
    if len(excerpt) < len(text) and not excerpt.endswith('...'):
        excerpt = excerpt.rstrip('.,;:!?') + '...'
    
    return excerpt


def _is_header_like(text: str) -> bool:
    """
    Checks if text looks like a header rather than content.
    
    Headers are typically:
    - All caps
    - End with colon
    - Very short (< 50 chars)
    - Start with numbers/bullets
    """
    text = text.strip()
    
    # Too short to be content
    if len(text) < 30:
        return True
    
    # All caps (likely a header)
    if text.isupper():
        return True
    
    # Ends with colon (likely a header)
    if text.endswith(':'):
        return True
    
    # Starts with bullet or number pattern
    if re.match(r'^[\d•\-\*\#\.\)]+\s', text):
        return True
    
    return False


def clean_text_for_display(text: str) -> str:
    """
    Cleans text for display in the UI.
    
    - Removes excessive whitespace
    - Removes control characters
    - Limits line breaks
    """
    if not text:
        return ""
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()
