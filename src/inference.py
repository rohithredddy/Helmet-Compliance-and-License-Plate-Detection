import os
import cv2
from pathlib import Path
from ultralytics import YOLO


COLOR_MAP = {
    "Helmet": (0, 255, 0),        # Green
    "Motorbike": (255, 0, 0),     # Blue
    "NoHelmet": (0, 0, 255),      # Red
    "PNumber": (245, 66, 188)     # Pink
}


class HelmetInference:

    def __init__(self, model_path, device=None):
        self.model = YOLO(model_path)
        self.device = device

    def detect_image(self, image_path, conf_threshold=0.25, save_path=None):
        """
        Run inference on a single image.
        Returns annotated image.
        Optionally saves output.
        """

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found: {image_path}")

        results = self.model(image_path, conf=conf_threshold, device=self.device)
        boxes = results[0].boxes

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = COLOR_MAP.get(class_name, (255, 255, 255))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )

            cv2.rectangle(img,
                          (x1, y1 - h - 8),
                          (x1 + w, y1),
                          color,
                          -1)

            cv2.putText(img,
                        label,
                        (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1)

        if save_path:
            cv2.imwrite(save_path, img)

        return img

    def detect_folder(self, folder_path, output_dir=None, conf_threshold=0.25):
        """
        Run inference on all images inside folder.
        Saves outputs if output_dir provided.
        """

        folder_path = Path(folder_path)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        valid_ext = [".jpg", ".jpeg", ".png"]

        for file in folder_path.iterdir():
            if file.suffix.lower() in valid_ext:

                save_path = None
                if output_dir:
                    save_path = output_dir / file.name

                self.detect_image(
                    str(file),
                    conf_threshold=conf_threshold,
                    save_path=str(save_path) if save_path else None
                )

        print("Inference completed.")