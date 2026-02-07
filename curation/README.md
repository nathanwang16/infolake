# Curation Module

Quality scoring and content curation algorithms with modular, protocol-based architecture.

## Architecture

The curation module has been refactored into a modular package with extensible components:

```
curation/
└── scoring/              # Modular scoring package
    ├── __init__.py       # Public API + backward compatibility
    ├── pipeline.py       # ScoringPipeline orchestrator
    ├── protocols.py      # Type protocols (ScoringMetric, ScoreAggregator)
    ├── registry.py       # MetricRegistry for extensibility
    ├── detection.py      # Content type detection
    ├── aggregation.py    # Weighted sigmoid + Wilson score
    ├── _compat.py        # Backward compatibility facade
    └── metrics/          # Individual metric modules
        ├── citation.py
        ├── depth.py
        ├── methodology.py
        ├── reputation.py
        ├── specificity.py
        ├── structure.py
        └── writing.py
```

## Usage

### Backward Compatible API

```python
# Old API still works via compatibility layer
from curation.scoring import scorer

score = scorer.score(document)  # Returns 0.0-1.0
```

### New Modular API

```python
from curation.scoring import ScoringPipeline

pipeline = ScoringPipeline()

# Step 1: Compute raw metrics
raw_metrics = pipeline.compute_raw_metrics(text, metadata)
# Returns: {'citation_quality': 0.8, 'depth': 0.6, ...}

# Step 2: Detect content type
content_type = pipeline.detect_content_type(text, metadata)
# Returns: 'scientific', 'technical_code', 'personal_essay', etc.

# Step 3: Aggregate with content-type-specific weights
quality_score = pipeline.compute_score(raw_metrics, content_type)

# Step 4: Compute Wilson score (sample-size-aware confidence)
wilson_score = pipeline.compute_document_wilson_score(raw_metrics)
```

## Built-in Metrics

| Metric | File | Measures |
|--------|------|----------|
| Citation Quality | `citation.py` | DOIs, reference density, academic citations |
| Content Depth | `depth.py` | Word count, paragraph structure, code blocks |
| Methodology | `methodology.py` | Methodology transparency, reproducibility |
| Reputation | `reputation.py` | Domain authority, author credibility |
| Specificity | `specificity.py` | Jargon density, technical depth |
| Structure | `structure.py` | Heading hierarchy, formatting quality |
| Writing Quality | `writing.py` | Grammar, typos, readability |

## Content-Type-Specific Scoring

Different content types use different weight profiles:

| Content Type | Key Metrics | Quality Threshold |
|--------------|-------------|-------------------|
| scientific | citation_quality (0.28), methodology (0.24) | 0.50 |
| technical_code | code_quality (0.27), recency (0.23) | 0.45 |
| personal_essay | writing_quality (0.32), specificity (0.26) | 0.40 |
| news | source_attribution (0.31), multiple_perspectives (0.24) | 0.50 |
| documentation | completeness (0.32), accuracy (0.26) | 0.45 |

## Quality Categories

```python
HIGH_QUALITY = score >= 0.7    # Top-tier content
MEDIUM_QUALITY = 0.4 <= score < 0.7  # Acceptable
LOW_QUALITY = score < 0.4     # Needs review
```

## Wilson Score

Sample-size-aware confidence scoring:

```python
# Compute Wilson score from raw metrics
wilson_score = pipeline.compute_document_wilson_score(raw_metrics)

# Or compute from counts
positive_signals = 50  # Number of positive quality signals
total_signals = 55     # Total signals evaluated
wilson = pipeline.compute_wilson_score(positive_signals, total_signals, z=1.96)

# Returns conservative lower bound of confidence interval
# Example: 50/55 = 90.9% raw score → 80.6% Wilson score (95% CI)
```

## Custom Metrics

Extend the system with custom metrics following the `ScoringMetric` protocol:

```python
from curation.scoring.protocols import ScoringMetric
from typing import List, Dict, Any

class MyCustomMetric:
    @property
    def name(self) -> str:
        return "custom_metric"

    def compute(
        self,
        text: str,
        words: List[str],
        sentences: List[str],
        metadata: Dict[str, Any],
    ) -> float:
        # Your metric logic here
        return 0.5  # Return 0.0-1.0

# Register and use
pipeline = ScoringPipeline()
pipeline.registry.register(MyCustomMetric())
```

## Content-Type-Specific Weight Calibration

Update weights for specific content types:

```python
pipeline.update_weights("scientific", {
    "citation_quality": 0.30,
    "methodology": 0.25,
    "depth": 0.20,
    "specificity": 0.15,
    "writing": 0.10,
})

# Weights are applied automatically based on detected content type
score = pipeline.compute_score(raw_metrics, content_type="scientific")
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
  },
  "calibration": {
    "current_version": 1,
    "holdout_fraction": 0.2,
    "default_content_type": "technical_code"
  }
}
```

## Golden Set Calibration

Weights are calibrated using a Golden Set of manually labeled documents:
- 200 exemplary documents per content type
- 200 garbage documents per content type
- Topic-cluster cross-validation (prevents 28-40% overfit)
- QADI metrics for validation (quantity vs allocation diagnosis)
