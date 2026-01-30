# Curation Module

Quality scoring and content curation algorithms.

## Components

### Scoring (`scoring.py`)
Multi-dimensional quality scoring:

```python
from curation.scoring import QualityScorer

scorer = QualityScorer()
score = scorer.score(document)  # Returns 0.0-1.0
```

## Scoring Dimensions

| Dimension | Weight | Signals |
|-----------|--------|---------|
| Content Depth | 0.35 | Word count, paragraph structure, code blocks |
| Authority | 0.25 | Domain reputation, author presence |
| Freshness | 0.15 | Publication date, update frequency |
| Originality | 0.15 | Dedup score, citation density |
| Accessibility | 0.10 | Reading level, formatting quality |

## Quality Categories

```python
HIGH_QUALITY = score >= 0.7    # Top-tier content
MEDIUM_QUALITY = 0.4 <= score < 0.7  # Acceptable
LOW_QUALITY = score < 0.4     # Needs review
```

## Wilson Score

For confidence-aware ranking with limited data:

```python
def wilson_score(successes, total, z=1.96):
    """Lower bound of Wilson score interval (95% CI)"""
    if total == 0:
        return 0
    p = successes / total
    return (p + z*z/(2*n) - z*sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)
```

## Configuration

```json
{
  "content_extraction": {
    "min_length": 100,
    "max_length": 100000
  },
  "language": {
    "target": "en",
    "use_langdetect": true
  }
}
```

## Future Enhancements

- Golden set calibration (see `golden_metrics/`)
- QADI metrics for alignment tracking
- Topic-cluster cross-validation
