import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class DocumentRegion:
    """Represents a detected region within a document.

    Attributes:
        label (str): The classification of the region (e.g., 'header', 'content').
        x (int): X-coordinate of the top-left corner.
        y (int): Y-coordinate of the top-left corner.
        w (int): Width of the region.
        h (int): Height of the region.
        conf (float): Confidence score of the detection.
    """
    label: str
    x: int
    y: int
    w: int
    h: int
    conf: float = 1.0

    def crop(self, img: np.ndarray) -> np.ndarray:
        """Crops this region from the provided image.

        Args:
            img: The source image array.

        Returns:
            np.ndarray: The cropped region array.
        """
        return img[self.y:self.y+self.h, self.x:self.x+self.w]

    def __repr__(self):
        return f"DocumentRegion(label='{self.label}', x={self.x}, y={self.y}, w={self.w}, h={self.h}, conf={self.conf:.2f})"


class DocumentRegionDetector:
    """Detects logical regions in a document image.

    This class provides functionality to segment a document into logical sections
    using either a trained YOLOv8 model or a rule-based contour fallback.

    Attributes:
        yolo (Optional[YOLO]): The YOLOv8 model instance.
    """

    def __init__(self, yolo_weights: Optional[str] = None):
        """Initializes the detector with optional YOLOv8 weights.

        Args:
            yolo_weights: Path to the .pt file for YOLOv8.
        """
        self.yolo = None

        if yolo_weights:
            try:
                from ultralytics import YOLO
                self.yolo = YOLO(yolo_weights)
                logger.info(f"Loaded YOLOv8 model from {yolo_weights}")
            except Exception as e:
                logger.warning(f"Failed to load YOLOv8: {str(e)}. Falling back to contour detection.")

    def detect_with_yolo(self, img: np.ndarray) -> List[DocumentRegion]:
        """Performs region detection using the YOLOv8 model.

        Args:
            img: Input image array.

        Returns:
            List[DocumentRegion]: List of detected regions.
        """
        if self.yolo is None:
            return []

        results = self.yolo(img, verbose=False)[0]
        regions = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = results.names[int(box.cls[0])]
            regions.append(DocumentRegion(
                label=label, x=x1, y=y1,
                w=x2-x1, h=y2-y1, conf=conf
            ))
        return regions

    def detect_with_contours(self, img: np.ndarray) -> List[DocumentRegion]:
        """Segments the document based on horizontal whitespace sections.

        Args:
            img: Input image array.

        Returns:
            List[DocumentRegion]: List of identified regions.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        h, w = gray.shape

        # Identify horizontal whitespace rows
        row_means = np.mean(gray, axis=1)
        is_white = row_means > 240
        regions = []

        in_content = False
        seg_start = 0
        min_seg_h = h // 20

        for row_idx in range(h):
            if not is_white[row_idx] and not in_content:
                in_content = True
                seg_start = row_idx
            elif is_white[row_idx] and in_content:
                seg_end = row_idx
                seg_h = seg_end - seg_start
                if seg_h > min_seg_h:
                    regions.append(DocumentRegion(
                        label=self._label_region(seg_start, h),
                        x=0, y=seg_start, w=w, h=seg_h,
                        conf=0.9
                    ))
                in_content = False

        if in_content and (h - seg_start) > min_seg_h:
            regions.append(DocumentRegion(
                label=self._label_region(seg_start, h),
                x=0, y=seg_start, w=w, h=h - seg_start,
                conf=0.9
            ))

        if not regions:
            logger.info("No distinct regions found; returning full document.")
            regions = [DocumentRegion(label="full_document", x=0, y=0, w=w, h=h, conf=1.0)]

        return regions

    def _label_region(self, y: int, total_h: int) -> str:
        """Assigns a classification label based on the vertical position.

        Args:
            y: Vertical starting coordinate.
            total_h: Total height of the document.

        Returns:
            str: The assigned region label.
        """
        ratio = y / total_h
        if ratio < 0.15:
            return "header"
        elif ratio < 0.35:
            return "patient_info"
        elif ratio < 0.75:
            return "content"
        else:
            return "footer"

    def detect(self, img: np.ndarray) -> List[DocumentRegion]:
        """High-level detection method that handles model fallback logic.

        Args:
            img: Input image array.

        Returns:
            List[DocumentRegion]: List of detected regions.
        """
        if self.yolo is not None:
            return self.detect_with_yolo(img)
        return self.detect_with_contours(img)

    def draw_regions(self, img: np.ndarray, regions: List[DocumentRegion]) -> np.ndarray:
        """Annotates the image with bounding boxes for visualization.

        Args:
            img: Image array to annotate.
            regions: List of regions to draw.

        Returns:
            np.ndarray: Annotated BGR image array.
        """
        vis = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        color_map: Dict[str, Tuple[int, int, int]] = {
            "header": (255, 0, 0),
            "patient_info": (0, 255, 0),
            "content": (0, 0, 255),
            "footer": (128, 128, 0),
            "full_document": (200, 200, 0)
        }

        for r in regions:
            color = color_map.get(r.label, (100, 100, 100))
            cv2.rectangle(vis, (r.x, r.y), (r.x+r.w, r.y+r.h), color, 3)
            label_text = f"{r.label} ({r.conf:.2f})"
            cv2.putText(vis, label_text, (r.x+5, r.y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return vis
