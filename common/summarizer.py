"""
Content summarization using small, fast LLM/ML models.

Provides concise summaries for document previews with multiple backends:
1. Local transformer models (DistilBART, T5-small) - fastest, no API needed
2. OpenAI API (gpt-4o-mini) - very fast, high quality
3. Ollama local models - if available
4. Fallback to algorithmic excerpt extraction

Usage:
    summarizer = Summarizer()
    summary = summarizer.summarize(text, max_length=100)
"""

import os
import re
import threading
from typing import Optional

from common.logging.logger import get_logger
from common.config import config

logger = get_logger("summarizer")

# Thread-safe model singleton
_model_lock = threading.Lock()
_summarizer_model = None
_summarizer_tokenizer = None
_model_device = None
_model_loaded = False
_model_failed = False


class Summarizer:
    """
    Fast, concise content summarizer using small ML models.
    
    Prioritizes speed while maintaining quality:
    1. Uses T5-small for local inference - fast (~600ms on MPS)
    2. Falls back to OpenAI gpt-4o-mini if API key configured
    3. Falls back to algorithmic excerpt if models unavailable
    """
    
    # T5-small is faster than BART variants and works well for summarization
    LOCAL_MODEL = "t5-small"
    
    def __init__(self):
        self.backend = self._detect_backend()
        self._openai_client = None
        
        logger.info(f"Summarizer initialized with backend: {self.backend}")
    
    def _detect_backend(self) -> str:
        """Detects available summarization backend."""
        # Check config preference
        preferred = config.get("summarizer.backend")
        
        if preferred == "openai":
            if os.environ.get("OPENAI_API_KEY") or config.get("llm.api_key"):
                return "openai"
        
        if preferred == "local" or preferred == "auto":
            # Check if transformers is available
            try:
                import transformers
                return "local"
            except ImportError:
                pass
        
        if preferred == "ollama" or preferred == "auto":
            # Check if ollama is available
            try:
                import requests
                resp = requests.get("http://localhost:11434/api/tags", timeout=1)
                if resp.status_code == 200:
                    return "ollama"
            except Exception:
                pass
        
        # Check OpenAI as fallback
        if os.environ.get("OPENAI_API_KEY") or config.get("llm.api_key"):
            return "openai"
        
        # Final fallback
        return "excerpt"
    
    def summarize(
        self,
        text: str,
        max_length: int = 80,
        min_length: int = 20
    ) -> Optional[str]:
        """
        Generates a concise summary of the text.
        
        Args:
            text: Full document text
            max_length: Maximum summary length in words
            min_length: Minimum summary length in words
            
        Returns:
            Concise summary string or None if failed
        """
        if not text or len(text.strip()) < 100:
            return None
        
        # Truncate very long texts for speed (first 2000 chars is usually enough)
        if len(text) > 2000:
            text = text[:2000]
        
        try:
            if self.backend == "local":
                return self._summarize_local(text, max_length, min_length)
            elif self.backend == "openai":
                return self._summarize_openai(text, max_length)
            elif self.backend == "ollama":
                return self._summarize_ollama(text, max_length)
            else:
                return self._summarize_excerpt(text, max_length)
        except Exception as e:
            logger.warning(f"Summarization failed ({self.backend}): {e}")
            # Fallback to excerpt
            return self._summarize_excerpt(text, max_length)
    
    def _summarize_local(
        self,
        text: str,
        max_length: int,
        min_length: int
    ) -> Optional[str]:
        """Uses local T5-small model for fast summarization (~600ms on MPS)."""
        global _summarizer_model, _summarizer_tokenizer, _model_device, _model_loaded, _model_failed
        
        if _model_failed:
            return self._summarize_excerpt(text, max_length)
        
        # Lazy load model (direct loading, not pipeline - works with transformers 5.x)
        if not _model_loaded:
            with _model_lock:
                if not _model_loaded and not _model_failed:
                    try:
                        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                        import torch
                        
                        logger.info(f"Loading summarization model: {self.LOCAL_MODEL}")
                        
                        _summarizer_tokenizer = AutoTokenizer.from_pretrained(self.LOCAL_MODEL)
                        _summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(self.LOCAL_MODEL)
                        
                        # Select device
                        device_config = config.get("embedding.device")
                        if device_config == "mps" and torch.backends.mps.is_available():
                            _model_device = "mps"
                        elif device_config == "cuda" and torch.cuda.is_available():
                            _model_device = "cuda"
                        else:
                            _model_device = "cpu"
                        
                        _summarizer_model = _summarizer_model.to(_model_device)
                        
                        # Warm-up run for consistent fast inference
                        logger.info(f"Warming up model on {_model_device}...")
                        warmup_text = "summarize: The field of artificial intelligence has seen remarkable advances. Machine learning algorithms now power recommendation systems and autonomous vehicles."
                        warmup_inputs = _summarizer_tokenizer(warmup_text, return_tensors="pt", max_length=256, truncation=True)
                        warmup_inputs = {k: v.to(_model_device) for k, v in warmup_inputs.items()}
                        _ = _summarizer_model.generate(**warmup_inputs, max_new_tokens=30)
                        
                        _model_loaded = True
                        logger.info("Summarization model loaded and warmed up")
                        
                    except Exception as e:
                        logger.warning(f"Failed to load local model: {e}")
                        _model_failed = True
                        return self._summarize_excerpt(text, max_length)
        
        if not _summarizer_model or not _summarizer_tokenizer:
            return self._summarize_excerpt(text, max_length)
        
        try:
            # T5 expects "summarize: " prefix
            input_text = f"summarize: {text}"
            
            # Tokenize
            inputs = _summarizer_tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            inputs = {k: v.to(_model_device) for k, v in inputs.items()}
            
            # Generate summary
            max_tokens = int(max_length * 1.3)
            outputs = _summarizer_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                min_length=min_length,
                do_sample=False,
                num_beams=1,  # Greedy for speed
            )
            
            summary = _summarizer_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._clean_summary(summary)
            
        except Exception as e:
            logger.warning(f"Local summarization failed: {e}")
            return self._summarize_excerpt(text, max_length)
    
    def _summarize_openai(self, text: str, max_length: int) -> Optional[str]:
        """Uses OpenAI gpt-4o-mini for fast summarization."""
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai package not installed")
            return self._summarize_excerpt(text, max_length)
        
        if not self._openai_client:
            api_key = os.environ.get("OPENAI_API_KEY") or config.get("llm.api_key")
            if not api_key:
                return self._summarize_excerpt(text, max_length)
            self._openai_client = OpenAI(api_key=api_key)
        
        try:
            response = self._openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"Summarize the following text in {max_length} words or less. Be concise and capture the main point. Output only the summary, nothing else."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            return self._clean_summary(summary)
            
        except Exception as e:
            logger.warning(f"OpenAI summarization failed: {e}")
            return self._summarize_excerpt(text, max_length)
    
    def _summarize_ollama(self, text: str, max_length: int) -> Optional[str]:
        """Uses Ollama local models for summarization."""
        try:
            import requests
        except ImportError:
            return self._summarize_excerpt(text, max_length)
        
        model = config.get("summarizer.ollama_model")
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": f"Summarize in {max_length} words or less. Be concise:\n\n{text}",
                    "stream": False,
                    "options": {
                        "num_predict": 150,
                        "temperature": 0.3
                    }
                },
                timeout=10
            )
            
            if response.status_code == 200:
                summary = response.json().get("response", "").strip()
                return self._clean_summary(summary)
                
        except Exception as e:
            logger.warning(f"Ollama summarization failed: {e}")
        
        return self._summarize_excerpt(text, max_length)
    
    def _summarize_excerpt(self, text: str, max_words: int) -> Optional[str]:
        """Fallback to algorithmic excerpt extraction."""
        from common.text_utils import extract_excerpt
        return extract_excerpt(text, max_words=max_words, prefer_first_paragraph=True)
    
    def _clean_summary(self, summary: str) -> str:
        """
        Cleans up generated summary and converts to concise phrase-style.
        """
        import re
        
        if not summary:
            return ""
        
        # Remove common prefixes from LLM outputs
        prefixes_to_remove = [
            r"^Here('s| is) a summary:?\s*",
            r"^Summary:?\s*",
            r"^The (text|article|document|content) (discusses?|describes?|explains?|is about|covers?)\s*",
            r"^This (text|article|document|content)\s*",
            r"^In (this|the) (text|article|document),?\s*",
        ]
        
        for pattern in prefixes_to_remove:
            summary = re.sub(pattern, '', summary, flags=re.IGNORECASE)
        
        # Remove junk patterns that shouldn't be in summaries
        junk_patterns = [
            r'(?i)for (more )?information,?\s*(please )?(see|visit|click).*$',
            r'(?i)for license information.*$',
            r'(?i)cookie (policy|consent|settings).*',
            r'(?i)privacy (policy|notice|statement).*',
            r'(?i)all rights reserved.*$',
            r'(?i)copyright \d{4}.*$',
            r'(?i)terms (of use|and conditions).*$',
            r'(?i)please (enable|verify|accept).*$',
            r'(?i)skip to (main )?content.*$',
        ]
        
        for pattern in junk_patterns:
            summary = re.sub(pattern, '', summary)
        
        # Remove quotes if wrapped
        if summary.startswith('"') and summary.endswith('"'):
            summary = summary[1:-1]
        
        # Clean up whitespace
        summary = re.sub(r'\s+', ' ', summary).strip()
        
        # Convert to concise phrase-style if too sentence-heavy
        if summary.count('.') > 2 or len(summary) > 180:
            summary = self._to_phrases(summary)
        
        # Clean up trailing punctuation
        summary = re.sub(r'[.,;:\s]+$', '', summary)
        
        return summary.strip()
    
    def _to_phrases(self, text: str) -> str:
        """Converts full sentences to concise key phrases."""
        import re
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        phrases = []
        for sent in sentences[:3]:  # Max 3 key points
            # Remove common sentence starters
            sent = re.sub(r'^(The|This|It|There|These|A|An)\s+', '', sent)
            sent = re.sub(r'^(is|are|was|were|has|have|had|being)\s+', '', sent)
            
            # Take first clause if too long
            if len(sent) > 70:
                parts = re.split(r'[,;:]', sent)
                sent = parts[0] if parts else sent[:70]
            
            sent = sent.strip().rstrip('.')
            if sent and len(sent) > 10:
                phrases.append(sent)
        
        return '; '.join(phrases[:3])


# Singleton instance
_summarizer_instance = None


def get_summarizer() -> Summarizer:
    """Returns singleton summarizer instance."""
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = Summarizer()
    return _summarizer_instance


def summarize_content(text: str, max_length: int = 80) -> Optional[str]:
    """Convenience function for summarization."""
    return get_summarizer().summarize(text, max_length=max_length)
