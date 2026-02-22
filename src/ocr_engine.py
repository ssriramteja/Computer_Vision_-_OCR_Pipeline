import pytesseract
import numpy as np
import logging
import re
from PIL import Image
from dataclasses import dataclass
from typing import List, Union, Dict

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Stores the result of an OCR operation for a specific document region.

    Attributes:
        region_label (str): The label assigned to the source region.
        raw_text (str): The literal text output from the OCR engine.
        confidence (float): Average word-level confidence percentage.
        word_count (int): Number of identified words in the text.
    """
    region_label: str
    raw_text: str
    confidence: float
    word_count: int

    def clean_text(self) -> str:
        """Sanitizes the OCR text by removing noise and normalizing whitespace.

        Returns:
            str: The cleaned and trimmed text.
        """
        # Remove non-alphanumeric/punctuation characters except for specific medical symbols
        text = re.sub(r"[^\w\s.,:\-/@#%()°+<>\[\]⚠]", " ", self.raw_text)
        # Normalize excessive whitespace
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()


class TesseractOCREngine:
    """Wrapper for the Tesseract OCR engine, optimized for medical document parsing.

    This engine supports multiple Page Segmentation Modes (PSM) and utilizes
    the LSTM-based OCR engine for higher accuracy.

    Attributes:
        lang (str): Language code for Tesseract (default: 'eng').
    """

    # Tesseract Page Segmentation Modes mapping
    PSM_MODES: Dict[str, int] = {
        "full_page": 3,      # Fully automatic page segmentation, but no OSD
        "single_block": 6,   # Assume a single uniform block of text
        "single_line": 7,    # Treat the image as a single text line
        "sparse_text": 11    # Find as much text as possible in no particular order
    }

    def __init__(self, lang: str = "eng"):
        """Initializes the OCR engine and verifies Tesseract installation.

        Args:
            lang: Target language code.

        Raises:
            RuntimeError: If the Tesseract binary is not found on the system.
        """
        self.lang = lang
        self._verify_installation()

    def _verify_installation(self):
        """Checks if Tesseract is correctly installed and accessible."""
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Verified Tesseract installation (version {version})")
        except Exception as e:
            logger.error(f"Tesseract not found: {str(e)}")
            raise RuntimeError(
                "Tesseract executable not found. Please install the Tesseract binary for your OS."
            )

    def _get_config(self, psm: int) -> str:
        """Constructs the Tesseract command-line configuration string.

        Args:
            psm: The Page Segmentation Mode integer.

        Returns:
            str: Formatted configuration string.
        """
        return (
            f"--psm {psm} "
            "--oem 3 "  # Use LSTM engine
            "-c tessedit_char_whitelist="
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            "0123456789 .,:-/@#%()°+<>[]⚠"
        )

    def ocr_image(
        self,
        img: Union[np.ndarray, Image.Image],
        region_label: str = "unknown",
        psm_mode: str = "full_page"
    ) -> OCRResult:
        """Performs OCR on the provided image array or PIL Image object.

        Args:
            img: Input image source.
            region_label: Label for contextual tracking of the result.
            psm_mode: The segmentation mode key from PSM_MODES.

        Returns:
            OCRResult: Structured output from the OCR operation.
        """
        # Convert numpy BGR or Grayscale to PIL RGB
        if isinstance(img, np.ndarray):
            if len(img.shape) == 2:
                pil_img = Image.fromarray(img)
            else:
                import cv2
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            pil_img = img

        psm = self.PSM_MODES.get(psm_mode, 3)
        config = self._get_config(psm)

        try:
            # Extract detailed data for confidence calculation
            data = pytesseract.image_to_data(
                pil_img, lang=self.lang, config=config,
                output_type=pytesseract.Output.DICT
            )
            raw_text = pytesseract.image_to_string(pil_img, lang=self.lang, config=config)
            
            # Confidence filtering (Tesseract returns -1 for empty/non-text areas)
            confs = [int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) >= 0]
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            word_count = len([w for w in data["text"] if w.strip()])

            return OCRResult(
                region_label=region_label,
                raw_text=raw_text,
                confidence=round(avg_conf, 2),
                word_count=word_count
            )
        except Exception as e:
            logger.error(f"OCR failed for region '{region_label}': {str(e)}")
            return OCRResult(region_label=region_label, raw_text="", confidence=0.0, word_count=0)

    def ocr_regions(
        self,
        full_img: np.ndarray,
        regions: List
    ) -> List[OCRResult]:
        """Iterates over detected document regions and performs targeted OCR.

        Args:
            full_img: The full document image source.
            regions: List of DocumentRegion objects.

        Returns:
            List[OCRResult]: Aggregated results from each region.
        """
        results = []
        logger.info(f"Initiating OCR on {len(regions)} document regions.")
        
        for region in regions:
            cropped = region.crop(full_img)

            # Context-aware PSM selection
            if region.label in ("header", "patient_info", "footer"):
                psm = "single_block"
            else:
                psm = "full_page"

            result = self.ocr_image(cropped, region_label=region.label, psm_mode=psm)
            results.append(result)

        return results
