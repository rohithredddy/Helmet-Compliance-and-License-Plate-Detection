"""
Data validation for YOLO-format datasets.
Checks for orphan files, invalid labels, and data integrity issues.
"""

import logging
from pathlib import Path

from config.settings import Settings
from data_preparation.dataset_loader import DatasetLoader

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate YOLO dataset integrity."""

    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir else Settings.DATA_DIR
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        self.loader = DatasetLoader(data_dir=self.data_dir)
        self.errors = []
        self.warnings = []

    def check_orphan_images(self, split):
        """Find images without corresponding label files."""
        img_dir = self.images_dir / split
        lbl_dir = self.labels_dir / split
        image_exts = set(Settings.IMAGE_EXTENSIONS)
        orphans = []

        if not img_dir.exists():
            return orphans

        for img in img_dir.iterdir():
            if img.suffix.lower() in image_exts:
                label_path = lbl_dir / (img.stem + ".txt")
                if not label_path.exists():
                    orphans.append(img.name)

        if orphans:
            self.warnings.append(
                f"[{split}] {len(orphans)} images without labels: {orphans[:5]}..."
            )
        return orphans

    def check_orphan_labels(self, split):
        """Find label files without corresponding images."""
        lbl_dir = self.labels_dir / split
        img_dir = self.images_dir / split
        orphans = []

        if not lbl_dir.exists():
            return orphans

        for lbl in lbl_dir.glob("*.txt"):
            found = False
            for ext in Settings.IMAGE_EXTENSIONS:
                if (img_dir / (lbl.stem + ext)).exists():
                    found = True
                    break
            if not found:
                orphans.append(lbl.name)

        if orphans:
            self.warnings.append(
                f"[{split}] {len(orphans)} labels without images: {orphans[:5]}..."
            )
        return orphans

    def check_label_format(self, split):
        """Validate label file format and class IDs."""
        lbl_dir = self.labels_dir / split
        invalid_files = []

        if not lbl_dir.exists():
            return invalid_files

        for lbl_file in lbl_dir.glob("*.txt"):
            with open(lbl_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if len(parts) < 5:
                        self.errors.append(
                            f"[{split}] {lbl_file.name}:{line_num} - Expected 5 values, got {len(parts)}"
                        )
                        invalid_files.append(lbl_file.name)
                        continue

                    class_id = int(parts[0])
                    if class_id < 0 or class_id >= Settings.NUM_CLASSES:
                        self.errors.append(
                            f"[{split}] {lbl_file.name}:{line_num} - Invalid class ID: {class_id}"
                        )

                    for i, val in enumerate(parts[1:5], 1):
                        v = float(val)
                        if v < 0.0 or v > 1.0:
                            self.errors.append(
                                f"[{split}] {lbl_file.name}:{line_num} - Bbox value out of [0,1]: {v}"
                            )

        return invalid_files

    def validate_all(self):
        """Run all validation checks across all splits."""
        self.errors = []
        self.warnings = []

        summary = {}
        for split in Settings.SPLITS:
            logger.info("Validating '%s' split...", split)
            orphan_imgs = self.check_orphan_images(split)
            orphan_lbls = self.check_orphan_labels(split)
            invalid = self.check_label_format(split)

            summary[split] = {
                "orphan_images": len(orphan_imgs),
                "orphan_labels": len(orphan_lbls),
                "invalid_labels": len(invalid),
            }

        # Print report
        logger.info("=" * 50)
        logger.info("VALIDATION REPORT")
        logger.info("=" * 50)

        for split, stats in summary.items():
            logger.info(
                "[%s] orphan_images=%d, orphan_labels=%d, invalid_labels=%d",
                split, stats["orphan_images"], stats["orphan_labels"], stats["invalid_labels"]
            )

        if self.errors:
            logger.error("ERRORS (%d):", len(self.errors))
            for err in self.errors[:20]:
                logger.error("  %s", err)
        else:
            logger.info("No errors found.")

        if self.warnings:
            logger.warning("WARNINGS (%d):", len(self.warnings))
            for w in self.warnings:
                logger.warning("  %s", w)

        is_valid = len(self.errors) == 0
        logger.info("Dataset valid: %s", is_valid)
        return is_valid, summary


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    validator = DataValidator()
    valid, report = validator.validate_all()
    sys.exit(0 if valid else 1)
