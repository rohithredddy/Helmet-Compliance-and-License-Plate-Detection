"""
Violation detection logic.
Identifies helmet violations from detection results.
"""

import logging
from config.settings import Settings

logger = logging.getLogger(__name__)


class ViolationAnalyzer:
    """Analyze detections for traffic violations."""

    def __init__(self, proximity_threshold=None):
        self.proximity_threshold = proximity_threshold or Settings.PROXIMITY_PIXELS

    def analyze(self, detections):
        """
        Analyze a list of detections for violations.

        Args:
            detections: list of detection dicts from HelmetDetector

        Returns:
            list of violation dicts with type, severity, and involved detections
        """
        violations = []

        # Check for no-helmet violations
        violations.extend(self._check_helmet_violations(detections))

        return violations

    def _check_helmet_violations(self, detections):
        """Flag NoHelmet detections as violations."""
        violations = []

        for i, det in enumerate(detections):
            if det["class_name"] == "NoHelmet":
                violations.append({
                    "type": "no_helmet",
                    "severity": "high",
                    "description": f"Rider without helmet detected (conf: {det['confidence']:.2f})",
                    "detection_indices": [i],
                    "confidence": det["confidence"],
                })

        return violations


    def get_violation_summary(self, violations):
        """Get a summary of violations by type."""
        summary = {}
        for v in violations:
            vtype = v["type"]
            if vtype not in summary:
                summary[vtype] = {"count": 0, "max_severity": "low"}
            summary[vtype]["count"] += 1

            severity_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
            if severity_rank.get(v["severity"], 0) > severity_rank.get(summary[vtype]["max_severity"], 0):
                summary[vtype]["max_severity"] = v["severity"]

        return summary
