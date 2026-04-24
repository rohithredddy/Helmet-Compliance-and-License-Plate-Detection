"""
Helmet detection inference engine.
Refactored from src/inference.py with structured output, in-memory support,
video processing, and Snowflake logging integration.
"""

import os
import time
import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from config.settings import Settings
from inference.violation_logic import ViolationAnalyzer

logger = logging.getLogger(__name__)


class HelmetDetector:
    """YOLOv8-based helmet and motorbike detection engine."""

    def __init__(self, model_path=None, device=None):
        model_path = model_path or str(Settings.DEFAULT_MODEL_PATH)
        self.model = YOLO(model_path)
        self.device = device
        self.violation_analyzer = ViolationAnalyzer()
        self.model_version = Path(model_path).parent.parent.name if model_path else "unknown"
        logger.info("Loaded model from: %s", model_path)

    def detect_image(self, image_input, conf_threshold=None, iou_threshold=None):
        """
        Run detection on a single image.

        Args:
            image_input: filepath (str), numpy array, or PIL Image
            conf_threshold: confidence threshold
            iou_threshold: IoU threshold for NMS

        Returns:
            dict with keys: detections, annotated_image, violations, processing_time_ms
        """
        conf = conf_threshold or Settings.CONFIDENCE_THRESHOLD
        iou = iou_threshold or Settings.IOU_THRESHOLD

        # Load image if path is provided
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Could not read image: {image_input}")
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
        else:
            # Assume PIL Image
            img = np.array(image_input)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        start_time = time.time()

        results = self.model(img, conf=conf, iou=iou, imgsz=Settings.IMAGE_SIZE, device=self.device)
        boxes = results[0].boxes

        detections = []
        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = self.model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })

        processing_time_ms = (time.time() - start_time) * 1000

        # Analyze violations
        violations = self.violation_analyzer.analyze(detections)

        # Draw annotations
        annotated = self._draw_annotations(img, detections, violations)

        return {
            "detections": detections,
            "annotated_image": annotated,
            "violations": violations,
            "processing_time_ms": processing_time_ms,
            "num_detections": len(detections),
            "num_violations": len(violations),
        }

    def detect_video(self, video_path, output_path=None, conf_threshold=None,
                     frame_skip=1, max_frames=None):
        """
        Run detection on a video file.

        Args:
            video_path: path to input video
            output_path: path for annotated output video (optional)
            conf_threshold: confidence threshold
            frame_skip: process every Nth frame
            max_frames: stop after N frames (None for all)

        Returns:
            list of per-frame results
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info("Video: %dx%d, %.1f fps, %d frames", width, height, fps, total_frames)

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_results = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_idx >= max_frames:
                break

            if frame_idx % frame_skip == 0:
                result = self.detect_image(frame, conf_threshold=conf_threshold)
                result["frame_index"] = frame_idx
                frame_results.append(result)

                if writer:
                    writer.write(result["annotated_image"])
            else:
                if writer:
                    writer.write(frame)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
            logger.info("Output video saved to: %s", output_path)

        logger.info("Processed %d frames, detected in %d frames", frame_idx, len(frame_results))
        return frame_results

    def _draw_annotations(self, img, detections, violations):
        """Draw bounding boxes and labels on the image."""
        annotated = img.copy()

        # Build a set of violation detection indices for highlighting
        violation_bboxes = set()
        for v in violations:
            for idx in v.get("detection_indices", []):
                violation_bboxes.add(idx)

        for i, det in enumerate(detections):
            class_name = det["class_name"]
            confidence = det["confidence"]
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

            color = Settings.COLOR_MAP.get(class_name, (255, 255, 255))

            # Thicker border for violations
            thickness = 3 if i in violation_bboxes else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            label = f"{class_name} {confidence:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - h - 8), (x1 + w, y1), color, -1)
            cv2.putText(
                annotated, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

        # Draw violation labels at the top
        y_offset = 30
        for v in violations:
            text = f"VIOLATION: {v['type']} (severity: {v['severity']})"
            cv2.putText(
                annotated, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )
            y_offset += 30

        return annotated
