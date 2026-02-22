import cv2
import numpy as np
import logging
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not installed. PDF support will be disabled.")


class ImagePreprocessor:
    """OpenCV-based preprocessing pipeline to maximize OCR accuracy.

    This class handles loading images (and PDFs), grayscale conversion,
    denoising, binarization, deskewing, and upscaling.

    Attributes:
        dpi (int): Dots per inch for PDF conversion.
        target_width (int): Target width for upscaling small images.
    """

    def __init__(self, dpi: int = 200, target_width: int = 2000):
        """Initializes the preprocessor with configuration.

        Args:
            dpi: DPI for PDF to image conversion.
            target_width: Minimum width for processed images.
        """
        self.dpi = dpi
        self.target_width = target_width

    def load(self, path: str) -> np.ndarray:
        """Loads an image or the first page of a PDF file.

        Args:
            path: Path to the image or PDF file.

        Returns:
            np.ndarray: The loaded image in BGR format.

        Raises:
            RuntimeError: If PDF support is missing or conversion fails.
            FileNotFoundError: If the file cannot be loaded.
        """
        ext = Path(path).suffix.lower()

        if ext == ".pdf":
            if not PDF2IMAGE_AVAILABLE:
                raise RuntimeError(
                    "pdf2image not installed. Please install it and poppler dependencies."
                )
            try:
                pages = convert_from_path(path, dpi=self.dpi)
                img = np.array(pages[0])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.error(f"Failed to convert PDF {path}: {str(e)}")
                raise RuntimeError(f"PDF conversion error: {str(e)}")
        else:
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Cannot load image: {path}")

        return img

    def to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Converts an image to grayscale.

        Args:
            img: BGR image array.

        Returns:
            np.ndarray: Grayscale image array.
        """
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        """Applies non-local means denoising to a grayscale image.

        Args:
            gray: Grayscale image array.

        Returns:
            np.ndarray: Denoised image array.
        """
        return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    def binarize(self, gray: np.ndarray) -> np.ndarray:
        """Applies adaptive thresholding to binarize the image.

        Args:
            gray: Grayscale image array.

        Returns:
            np.ndarray: Binary image array.
        """
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10
        )

    def deskew(self, binary: np.ndarray) -> np.ndarray:
        """Detects and corrects small rotation skews in a binary image.

        Args:
            binary: Binary image array.

        Returns:
            np.ndarray: Deskewed binary image array.
        """
        coords = np.column_stack(np.where(binary < 128))
        if coords.shape[0] < 100:
            return binary

        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Only correct skews within a +/- 15 degree range
        if abs(angle) > 15:
            return binary

        h, w = binary.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            binary, rotation_matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    def upscale(self, img: np.ndarray) -> np.ndarray:
        """Upscales an image if its width is below the target threshold.

        Args:
            img: Image array.

        Returns:
            np.ndarray: Upscaled (or original) image array.
        """
        h, w = img.shape[:2]
        if w < self.target_width:
            scale = self.target_width / w
            img = cv2.resize(
                img, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC
            )
        return img

    def process(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Executes the full preprocessing pipeline.

        Args:
            path: Path to the input file.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (original_bgr, processed_binary)
        """
        logger.info(f"Processing image: {path}")
        original = self.load(path)
        gray = self.to_grayscale(original)
        denoised = self.denoise(gray)
        binary = self.binarize(denoised)
        deskewed = self.deskew(binary)
        processed = self.upscale(deskewed)

        return original, processed

    def save(self, img: np.ndarray, out_path: str) -> None:
        """Saves an image to the specified path.

        Args:
            img: Image array to save.
            out_path: Output file path.
        """
        cv2.imwrite(out_path, img)
        logger.info(f"Saved processed image to: {out_path}")

    def to_pil(self, img: np.ndarray) -> Image.Image:
        """Converts a numpy array image to a PIL Image.

        Args:
            img: Image array.

        Returns:
            Image.Image: PIL Image object.
        """
        if len(img.shape) == 2:
            return Image.fromarray(img)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image Preprocessing Demo")
    parser.add_argument("--input", type=str, default="data/raw_images/prescription_001.pdf", help="Input file path")
    parser.add_argument("--output", type=str, default="data/raw_images/prescription_001_processed.png", help="Output file path")
    args = parser.parse_args()

    preprocessor = ImagePreprocessor()
    if Path(args.input).exists():
        try:
            original_img, processed_img = preprocessor.process(args.input)
            logger.info(f"Original shape: {original_img.shape}")
            logger.info(f"Processed shape: {processed_img.shape}")
            preprocessor.save(processed_img, args.output)
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
    else:
        logger.error(f"Input file not found: {args.input}. Please generate docs first.")
