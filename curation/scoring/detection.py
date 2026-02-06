"""Content type detection."""

from typing import Dict, Any

from common.config import config
from curation.scoring.protocols import ContentTypeDetector


class RuleBasedContentTypeDetector(ContentTypeDetector):
    """Rule-based content type detection extracted from PostProcessor."""

    CODE_INDICATORS = ['documentation', 'docs', 'api', 'reference', 'tutorial']
    ACADEMIC_INDICATORS = ['arxiv', 'journal', 'research', 'paper', 'study']
    NEWS_INDICATORS = ['news', 'article', 'breaking', 'report']

    def detect(self, text: str, metadata: Dict[str, Any]) -> str:
        url = (metadata.get('url') or '').lower()
        title = (metadata.get('title') or '').lower()

        if any(ind in url or ind in title for ind in self.CODE_INDICATORS):
            return 'technical_code'

        if any(ind in url or ind in title for ind in self.ACADEMIC_INDICATORS):
            return 'academic'

        if any(ind in url or ind in title for ind in self.NEWS_INDICATORS):
            return 'news'

        return config.get("calibration.default_content_type")
