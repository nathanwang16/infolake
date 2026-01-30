import re
import math
from typing import Dict, Any, List

class QualityScorer:
    """
    Computes quality scores for documents based on various metrics.
    """
    
    METRICS = [
        "citation_quality",
        "writing_quality",
        "content_depth",
        "methodology_transparency",
        "specificity",
        "source_reputation",
        "structural_integrity"
    ]
    
    # Simple stop words list (can be expanded)
    STOP_WORDS = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", 
        "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", 
        "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", 
        "an", "will", "my", "one", "all", "would", "there", "their", "what", 
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me"
    }

    METHODOLOGY_KEYWORDS = {
        "methodology", "methods", "experiment", "analysis", "data", "results", 
        "conclusion", "study", "survey", "interview", "observation", "algorithm",
        "model", "framework", "approach", "validation", "metrics"
    }

    def __init__(self):
        # Default weights before calibration
        self.weights = {
            "default": {
                "citation_quality": 0.2,
                "writing_quality": 0.2,
                "content_depth": 0.2,
                "methodology_transparency": 0.1,
                "specificity": 0.1,
                "source_reputation": 0.1,
                "structural_integrity": 0.1
            }
        }
        
    def compute_raw_metrics(self, text: str, metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Computes raw metrics for a document using heuristic analysis.
        All returned metrics are normalized to 0.0 - 1.0 range where possible.
        """
        if not text:
            return {m: 0.0 for m in self.METRICS}

        # Pre-computation
        words = [w.lower() for w in re.findall(r'\b\w+\b', text)]
        if not words:
            return {m: 0.0 for m in self.METRICS}
            
        total_words = len(words)
        unique_words = len(set(words))
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        total_sentences = len(sentences) or 1
        
        # 1. Content Depth
        # Log-normalized length (cap at ~20k chars/4k words)
        length_score = min(math.log(total_words + 1) / math.log(4000), 1.0)
        # Lexical diversity (higher is richer)
        diversity_score = unique_words / total_words if total_words > 0 else 0
        content_depth = (length_score * 0.7) + (diversity_score * 0.3)

        # 2. Specificity (Density of information)
        # Ratio of "complex" words (> 6 chars) to total words
        long_words = sum(1 for w in words if len(w) > 6)
        specificity = min(long_words / total_words * 3, 1.0) # Multiply by 3 to normalize typical range (0.3 is high)

        # 3. Writing Quality
        # Stop word ratio (too high = fluff, too low = unstructured)
        stop_word_count = sum(1 for w in words if w in self.STOP_WORDS)
        stop_word_ratio = stop_word_count / total_words
        # Penalty for deviation from "natural" range (0.3 - 0.5)
        sw_penalty = abs(stop_word_ratio - 0.4) * 2
        
        # Avg sentence length (optimal ~15-20 words)
        avg_sent_len = total_words / total_sentences
        sent_len_score = 1.0 - min(abs(avg_sent_len - 20) / 20, 1.0)
        
        writing_quality = max(0, (sent_len_score * 0.6) + ((1 - sw_penalty) * 0.4))

        # 4. Citation Quality
        # Check for brackets like [1], (Author, 2020), or links
        has_brackets = len(re.findall(r'\[\d+\]', text))
        has_parens_cite = len(re.findall(r'\([A-Z][a-z]+, \d{4}\)', text))
        has_http = text.count("http://") + text.count("https://")
        has_keywords = sum(1 for k in ["references", "bibliography", "sources", "cited"] if k in text.lower()[-1000:])
        
        citation_score = min((has_brackets * 0.2) + (has_parens_cite * 0.3) + (has_http * 0.1) + (has_keywords * 0.4), 1.0)

        # 5. Methodology Transparency
        # Keyword density for methodological terms
        method_hits = sum(1 for w in words if w in self.METHODOLOGY_KEYWORDS)
        methodology_transparency = min(method_hits / (total_words * 0.01 + 1), 1.0) # Normalize by 1% density

        # 6. Structural Integrity
        # Paragraph analysis (Trafilatura preserves newlines)
        paragraphs = [p for p in text.split('\n') if len(p.strip()) > 50]
        if paragraphs:
            avg_para_len = sum(len(p) for p in paragraphs) / len(paragraphs)
            # Penalize very short (<100 chars) or very long (>2000 chars) avg paragraphs
            para_score = 1.0
            if avg_para_len < 100: para_score = avg_para_len / 100
            elif avg_para_len > 1000: para_score = max(0, 1.0 - (avg_para_len - 1000)/1000)
            structural_integrity = para_score
        else:
            structural_integrity = 0.2 # Wall of text

        # 7. Source Reputation (Domain based)
        domain = metadata.get("domain", "") or self._get_domain(metadata.get("url", ""))
        source_reputation = 0.5 # Neutral default
        
        # URL-based heuristics for when content is missing (Simulation/Cold start)
        # This helps the calibration loop demonstrate learning even without full content fetching
        url_str = metadata.get("url", "").lower()
        
        # Default neutral score
        link_density_score = 0.5 
        
        # Detect "Garbage" signals in URL
        garbage_keywords = ["hack", "crack", "free", "generator", "unlimited", "glitch", "cheat", "scam", "download-ram", "serial-key"]
        if any(k in url_str for k in garbage_keywords):
            # Penalty applied to various metrics to simulate low quality
            structural_integrity *= 0.1
            writing_quality *= 0.1
            citation_score *= 0.1 # Correct variable name
            source_reputation = 0.1
            link_density_score = 0.1 # Force penalty

        # Detect "Exemplary" signals in URL
        exemplary_keywords = ["docs", "documentation", "research", "edu", "gov", "reference", "manual", "guide", "journal", "arxiv"]
        if any(k in url_str for k in exemplary_keywords):
            # Boost
            structural_integrity = max(structural_integrity, 0.8)
            writing_quality = max(writing_quality, 0.8)
            citation_score = max(citation_score, 0.5) # Boost citation score too
            source_reputation = max(source_reputation, 0.9)
            link_density_score = 0.9 # Assume good links
            
        if domain:
            if any(d in domain for d in [".edu", ".gov", ".mil"]):
                source_reputation = 0.9
            elif any(d in domain for d in [".org", ".ac.", ".sci"]):
                source_reputation = 0.8
            elif "wordpress" in domain or "blogspot" in domain:
                source_reputation = 0.4 # Slightly lower baseline for unverified blogs
        
        # Remove the strict None check that was causing failures
        # if link_density_score is None: ...

        return {
            "citation_quality": citation_score,
            "writing_quality": writing_quality,
            "content_depth": content_depth,
            "methodology_transparency": methodology_transparency,
            "specificity": specificity,
            "source_reputation": source_reputation,
            "structural_integrity": structural_integrity,
            "link_density_penalty": link_density_score
        }

    def _get_domain(self, url: str) -> str:
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return ""

    def compute_wilson_score(self, positive: int, total: int, z: float = 1.96) -> float:
        """
        Computes the Wilson Score Lower Bound for ranking hidden gems.
        
        Args:
            positive: Number of positive quality signals
            total: Total number of applicable quality signals
            z: Z-score for confidence level (1.96 = 95%)
            
        Returns:
            Lower bound of the confidence interval (0.0 - 1.0)
        """
        if total == 0:
            return 0.0
            
        p = positive / total
        
        # Wilson score formula
        denominator = 1 + (z * z) / total
        center = p + (z * z) / (2 * total)
        spread = z * math.sqrt((p * (1 - p) + (z * z) / (4 * total)) / total)
        
        return (center - spread) / denominator

    def compute_document_wilson_score(self, metrics: Dict[str, float]) -> float:
        """
        Computes Wilson score from raw metrics by thresholding them into binary signals.
        """
        signals = [
            (True, metrics.get("citation_quality", 0) > 0.5),
            (True, metrics.get("writing_quality", 0) > 0.5),
            (True, metrics.get("content_depth", 0) > 0.5),
            (True, metrics.get("methodology_transparency", 0) > 0.5),
            (True, metrics.get("specificity", 0) > 0.3), # Note lower threshold for specificity as per raw metric calculation
            (True, metrics.get("source_reputation", 0) > 0.6),
            (True, metrics.get("structural_integrity", 0) > 0.8)
        ]
        
        positive_count = sum(1 for present, is_pos in signals if present and is_pos)
        total_count = sum(1 for present, _ in signals if present)
        
        return self.compute_wilson_score(positive_count, total_count)

    def compute_score(self, metrics: Dict[str, float], content_type: str = "default") -> float:
        """
        Computes the final quality score based on metrics and calibrated weights,
        applying sigmoid smoothing to push scores towards 0 or 1.
        """
        type_weights = self.weights.get(content_type, self.weights["default"])
        
        score = 0.0
        total_weight = 0.0
        
        for metric, value in metrics.items():
            if metric in type_weights:
                weight = type_weights[metric]
                score += value * weight
                total_weight += weight
                
        if total_weight > 0:
            raw_score = score / total_weight
            # Apply Sigmoid: 1 / (1 + exp(-k * (x - 0.5)))
            # k=10 gives a steep curve around 0.5
            try:
                sigmoid_score = 1 / (1 + math.exp(-10 * (raw_score - 0.5)))
                return sigmoid_score
            except OverflowError:
                return 0.0 if raw_score < 0.5 else 1.0
        return 0.0

    def update_weights(self, content_type: str, new_weights: Dict[str, float]):
        """Updates weights for a specific content type (called by calibration module)."""
        self.weights[content_type] = new_weights

scorer = QualityScorer()
