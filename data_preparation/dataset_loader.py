"""
Dataset loader for YOLO-format datasets.
Parses images and label files into structured DataFrames.
"""

import os
import logging
from pathlib import Path
import pandas as pd
import cv2
import yaml

from config.settings import Settings

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load and parse YOLO-format dataset into structured data."""

    def __init__(self, data_dir=None, data_yaml=None):
        self.data_dir = Path(data_dir) if data_dir else Settings.DATA_DIR
        self.data_yaml = Path(data_yaml) if data_yaml else Settings.DATA_YAML
        self.class_names = Settings.CLASS_NAMES
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"

    def load_yaml_config(self):
        """Load and return the data.yaml configuration."""
        with open(self.data_yaml, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info("Loaded data.yaml: %d classes - %s", config.get("nc", 0), config.get("names", []))
        return config

    def parse_label_file(self, label_path):
        """
        Parse a single YOLO label file.
        Returns list of dicts with class_id, class_name, and bbox coords.
        """
        annotations = []
        if not os.path.exists(label_path):
            return annotations

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"unknown_{class_id}"

                    annotations.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "x_center": x_center,
                        "y_center": y_center,
                        "bbox_width": width,
                        "bbox_height": height,
                    })
        return annotations

    def get_image_info(self, image_path):
        """Get image dimensions and file size."""
        info = {
            "file_size_bytes": os.path.getsize(image_path),
            "image_width": None,
            "image_height": None,
        }
        try:
            img = cv2.imread(str(image_path))
            if img is not None:
                info["image_height"], info["image_width"] = img.shape[:2]
        except Exception as e:
            logger.warning("Could not read image %s: %s", image_path, e)
        return info

    def load_split(self, split):
        """
        Load all images and labels for a given split (train/val/test).
        Returns a DataFrame with image-level metadata and annotation counts.
        """
        images_path = self.images_dir / split
        labels_path = self.labels_dir / split

        if not images_path.exists():
            logger.warning("Images directory not found: %s", images_path)
            return pd.DataFrame()

        records = []
        image_extensions = set(Settings.IMAGE_EXTENSIONS)

        for img_file in sorted(images_path.iterdir()):
            if img_file.suffix.lower() not in image_extensions:
                continue

            label_file = labels_path / (img_file.stem + ".txt")
            annotations = self.parse_label_file(label_file)
            img_info = self.get_image_info(img_file)

            class_counts = {name: 0 for name in self.class_names}
            for ann in annotations:
                name = ann["class_name"]
                if name in class_counts:
                    class_counts[name] += 1

            records.append({
                "image_filename": img_file.name,
                "split": split,
                "image_width": img_info["image_width"],
                "image_height": img_info["image_height"],
                "file_size_bytes": img_info["file_size_bytes"],
                "num_objects": len(annotations),
                "helmet_count": class_counts.get("Helmet", 0),
                "nohelmet_count": class_counts.get("NoHelmet", 0),
                "motorbike_count": class_counts.get("Motorbike", 0),
                "pnumber_count": class_counts.get("PNumber", 0),
            })

        df = pd.DataFrame(records)
        logger.info("Loaded %d images from '%s' split", len(df), split)
        return df

    def load_all_splits(self):
        """Load all splits and return a combined DataFrame."""
        dfs = []
        for split in Settings.SPLITS:
            df = self.load_split(split)
            if not df.empty:
                dfs.append(df)

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            logger.info("Total dataset: %d images across %d splits", len(combined), len(dfs))
            return combined
        return pd.DataFrame()

    def load_all_annotations(self, split=None):
        """
        Load all individual annotations (bounding boxes) as a DataFrame.
        Each row is one bounding box annotation.
        """
        splits = [split] if split else Settings.SPLITS
        all_annotations = []

        for s in splits:
            labels_path = self.labels_dir / s
            images_path = self.images_dir / s

            if not labels_path.exists():
                continue

            for label_file in sorted(labels_path.glob("*.txt")):
                image_name = None
                for ext in Settings.IMAGE_EXTENSIONS:
                    candidate = images_path / (label_file.stem + ext)
                    if candidate.exists():
                        image_name = candidate.name
                        break

                annotations = self.parse_label_file(label_file)
                for ann in annotations:
                    ann["image_filename"] = image_name or label_file.stem
                    ann["split"] = s
                    all_annotations.append(ann)

        df = pd.DataFrame(all_annotations)
        logger.info("Loaded %d annotations", len(df))
        return df
