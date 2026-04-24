"""
Data splitter for creating or re-splitting train/val/test sets.
Used during retraining when new data is merged with existing data.
"""

import os
import shutil
import random
import logging
from pathlib import Path

from config.settings import Settings

logger = logging.getLogger(__name__)


class DataSplitter:
    """Split images and labels into train/val/test directories."""

    def __init__(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def collect_image_label_pairs(self, images_dir, labels_dir):
        """Find matching image-label file pairs."""
        pairs = []
        image_extensions = set(Settings.IMAGE_EXTENSIONS)

        for img_file in sorted(Path(images_dir).iterdir()):
            if img_file.suffix.lower() not in image_extensions:
                continue
            label_file = Path(labels_dir) / (img_file.stem + ".txt")
            if label_file.exists():
                pairs.append((img_file, label_file))
            else:
                logger.warning("No label for image: %s", img_file.name)

        logger.info("Found %d image-label pairs", len(pairs))
        return pairs

    def split_pairs(self, pairs):
        """Randomly split pairs into train/val/test sets."""
        random.seed(self.seed)
        shuffled = pairs.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_pairs = shuffled[:n_train]
        val_pairs = shuffled[n_train:n_train + n_val]
        test_pairs = shuffled[n_train + n_val:]

        logger.info(
            "Split: train=%d, val=%d, test=%d",
            len(train_pairs), len(val_pairs), len(test_pairs)
        )
        return {
            "train": train_pairs,
            "val": val_pairs,
            "test": test_pairs,
        }

    def execute_split(self, source_images_dir, source_labels_dir, output_dir):
        """
        Perform the split and copy files into output directory structure:
        output_dir/images/{train,val,test}/
        output_dir/labels/{train,val,test}/
        """
        pairs = self.collect_image_label_pairs(source_images_dir, source_labels_dir)
        splits = self.split_pairs(pairs)

        for split_name, split_pairs in splits.items():
            img_out = Path(output_dir) / "images" / split_name
            lbl_out = Path(output_dir) / "labels" / split_name
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            for img_file, lbl_file in split_pairs:
                shutil.copy2(str(img_file), str(img_out / img_file.name))
                shutil.copy2(str(lbl_file), str(lbl_out / lbl_file.name))

            logger.info("Copied %d pairs to %s", len(split_pairs), split_name)

        logger.info("Data split complete. Output: %s", output_dir)
        return splits
