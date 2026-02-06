"""Writing quality metric."""

from typing import Dict, Any, List

from curation.scoring.protocols import ScoringMetric

STOP_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
}


class WritingQualityMetric(ScoringMetric):
    """Scores writing quality via stop-word ratio and sentence length."""

    @property
    def name(self) -> str:
        return "writing_quality"

    @property
    def default_weight(self) -> float:
        return 0.2

    def compute(
        self,
        text: str,
        words: List[str],
        sentences: List[str],
        metadata: Dict[str, Any],
    ) -> float:
        total_words = len(words)
        total_sentences = len(sentences) or 1

        stop_word_count = sum(1 for w in words if w in STOP_WORDS)
        stop_word_ratio = stop_word_count / total_words
        sw_penalty = abs(stop_word_ratio - 0.4) * 2

        avg_sent_len = total_words / total_sentences
        sent_len_score = 1.0 - min(abs(avg_sent_len - 20) / 20, 1.0)

        return max(0, (sent_len_score * 0.6) + ((1 - sw_penalty) * 0.4))
