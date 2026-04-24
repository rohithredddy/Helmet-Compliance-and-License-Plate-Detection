"""
Batch inference processor.
Runs detection on all images in a folder with result logging.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings
from inference.detector import HelmetDetector
from inference.result_logger import ResultLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_batch_inference(
    input_dir,
    output_dir=None,
    model_path=None,
    conf_threshold=None,
    log_to_snowflake=True,
):
    """
    Run inference on all images in a directory.

    Args:
        input_dir: directory containing images
        output_dir: directory for annotated outputs (optional)
        model_path: path to model weights
        conf_threshold: confidence threshold
        log_to_snowflake: whether to log results to Snowflake

    Returns:
        list of result dicts
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    detector = HelmetDetector(model_path=model_path)
    result_logger = ResultLogger() if log_to_snowflake else None

    image_files = [
        f for f in sorted(input_dir.iterdir())
        if f.suffix.lower() in Settings.IMAGE_EXTENSIONS
    ]

    video_files = [
        f for f in sorted(input_dir.iterdir())
        if f.suffix.lower() in Settings.VIDEO_EXTENSIONS
    ]

    logger.info("Found %d images and %d videos in %s", len(image_files), len(video_files), input_dir)

    all_results = []

    # Process images
    for img_file in image_files:
        logger.info("Processing: %s", img_file.name)
        try:
            result = detector.detect_image(str(img_file), conf_threshold=conf_threshold)

            if output_dir:
                import cv2
                out_path = output_dir / img_file.name
                cv2.imwrite(str(out_path), result["annotated_image"])

            if result_logger:
                try:
                    result_logger.log_inference(
                        image_name=img_file.name,
                        result=result,
                        source_type="batch",
                    )
                except Exception as e:
                    logger.warning("Failed to log to Snowflake: %s", e)

            all_results.append({
                "filename": img_file.name,
                "type": "image",
                **{k: v for k, v in result.items() if k != "annotated_image"},
            })

        except Exception as e:
            logger.error("Failed to process %s: %s", img_file.name, e)

    # Process videos
    for vid_file in video_files:
        logger.info("Processing video: %s", vid_file.name)
        try:
            out_path = str(output_dir / vid_file.name) if output_dir else None
            frame_results = detector.detect_video(
                str(vid_file),
                output_path=out_path,
                conf_threshold=conf_threshold,
            )

            if result_logger and frame_results:
                try:
                    # Log summary for the video
                    total_detections = sum(r["num_detections"] for r in frame_results)
                    total_violations = sum(r["num_violations"] for r in frame_results)
                    total_time = sum(r["processing_time_ms"] for r in frame_results)

                    summary_result = {
                        "num_detections": total_detections,
                        "num_violations": total_violations,
                        "processing_time_ms": total_time,
                        "detections": [],
                        "violations": [],
                    }

                    result_logger.log_inference(
                        image_name=vid_file.name,
                        result=summary_result,
                        source_type="video",
                    )
                except Exception as e:
                    logger.warning("Failed to log video to Snowflake: %s", e)

            all_results.append({
                "filename": vid_file.name,
                "type": "video",
                "frames_processed": len(frame_results),
            })

        except Exception as e:
            logger.error("Failed to process video %s: %s", vid_file.name, e)

    logger.info("Batch inference complete. Processed %d files.", len(all_results))
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch inference for helmet detection")
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--model", default=None, help="Model weights path")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--no-snowflake", action="store_true", help="Disable Snowflake logging")

    args = parser.parse_args()

    results = run_batch_inference(
        input_dir=args.input,
        output_dir=args.output,
        model_path=args.model,
        conf_threshold=args.conf,
        log_to_snowflake=not args.no_snowflake,
    )
