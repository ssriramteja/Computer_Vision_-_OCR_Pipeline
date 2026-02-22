import json
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

from preprocessor import ImagePreprocessor
from detector import DocumentRegionDetector, DocumentRegion
from ocr_engine import TesseractOCREngine, OCRResult
from nlp_extractor import MedicalNLPExtractor, ExtractedDocument

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Consolidated result of the full OCR and NLP processing pipeline.

    Attributes:
        file_path (str): Path to the processed source file.
        doc_type (str): Classified document classification.
        extracted (ExtractedDocument): Structured data extracted from the text.
        full_ocr_text (str): Aggregate raw text from all document regions.
        num_regions (int): Number of regions identified in the document.
        avg_ocr_conf (float): Mean OCR confidence score across regions.
        processing_ms (float): Total processing time in milliseconds.
    """
    file_path: str
    doc_type: str
    extracted: ExtractedDocument
    full_ocr_text: str
    num_regions: int
    avg_ocr_conf: float
    processing_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Converts the pipeline result into a serializable dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the result.
        """
        return {
            "file_path": self.file_path,
            "doc_type": self.doc_type,
            "extracted": self.extracted.to_dict(),
            "full_ocr_text": self.full_ocr_text,
            "num_regions": self.num_regions,
            "avg_ocr_conf": self.avg_ocr_conf,
            "processing_ms": self.processing_ms
        }


class OCRPipeline:
    """Orchestrates the end-to-end medical document processing flow.

    Pipeline sequence:
    1. Preprocessing (OpenCV)
    2. Region Detection (Rule-based/ML)
    3. Optical Character Recognition (Tesseract)
    4. Natural Language Processing (spaCy/Regex Extraction)
    """

    def __init__(self):
        """Initializes all pipeline components."""
        logger.info("Initializing OCR pipeline components...")
        self.preprocessor = ImagePreprocessor()
        self.detector = DocumentRegionDetector()
        self.ocr = TesseractOCREngine()
        self.nlp = MedicalNLPExtractor()
        logger.info("OCR Pipeline initialized successfully.")

    def process(self, file_path: str) -> PipelineResult:
        """Executes the full pipeline on a single document file.

        Args:
            file_path: Path to the input image or PDF.

        Returns:
            PipelineResult: The aggregated output of the pipeline.
        """
        start_time = time.time()
        logger.info(f"Starting processing for file: {file_path}")

        # Step 1: Image Preprocessing
        original, processed = self.preprocessor.process(file_path)
        logger.info("Preprocessing complete.")

        # Step 2: Logical Region Detection
        regions = self.detector.detect(processed)
        logger.info(f"Detected {len(regions)} document regions.")

        # Step 3: Targeted OCR
        ocr_results = self.ocr.ocr_regions(processed, regions)
        full_text = "\n\n".join(r.raw_text for r in ocr_results if r.raw_text.strip())
        avg_conf = (
            sum(r.confidence for r in ocr_results) / len(ocr_results)
            if ocr_results else 0.0
        )
        logger.info(f"OCR execution complete (Average Confidence: {avg_conf:.2f}%).")

        # Step 4: NLP Clinical Extraction
        extracted_data = self.nlp.extract(full_text)
        logger.info(f"NLP extraction successful. Document type: {extracted_data.doc_type}")

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return PipelineResult(
            file_path=file_path,
            doc_type=extracted_data.doc_type,
            extracted=extracted_data,
            full_ocr_text=full_text,
            num_regions=len(regions),
            avg_ocr_conf=avg_conf,
            processing_ms=elapsed_ms
        )


def main():
    """CLI entry point for the OCR pipeline."""
    parser = argparse.ArgumentParser(description="Medical Document OCR and NLP Pipeline")
    parser.add_argument("input", type=str, nargs="?", help="Path to input image or PDF file")
    parser.add_argument("--output", type=str, help="Path to save result JSON")
    parser.add_argument("--demo", action="store_true", help="Run a demo over synthetic images")
    
    args = parser.parse_args()
    pipeline = OCRPipeline()

    if args.demo:
        test_files = [
            "data/raw_images/prescription_001.pdf",
            "data/raw_images/lab_report_001.pdf",
            "data/raw_images/discharge_summary_001.pdf",
        ]
        for f in test_files:
            path = Path(f) if Path(f).exists() else Path("..") / f
            if not path.exists():
                logger.warning(f"Skipping {f}: File not found.")
                continue
            
            result = pipeline.process(str(path))
            print_result(result)
            save_result(result)
    elif args.input:
        if not Path(args.input).exists():
            logger.error(f"Input file not found: {args.input}")
            return

        result = pipeline.process(args.input)
        print_result(result)
        save_result(result, args.output)
    else:
        parser.print_help()


def print_result(result: PipelineResult):
    """Prints a formatted summary of the processing result to the console."""
    print("\n" + "="*60)
    print(f" PROCESSING RESULT: {result.doc_type}")
    print("="*60)
    ext = result.extracted
    print(f"  Patient:  {ext.patient_name or 'N/A'}")
    print(f"  Doctor:   {ext.doctor_name or 'N/A'}")
    print(f"  Facility: {ext.hospital or 'N/A'}")
    print(f"  Date:     {ext.date or 'N/A'}")
    print(f"  MRN:      {ext.mrn or 'N/A'}")
    print(f"  Meds:     {len(ext.medications)} items identified")
    print(f"  Labs:     {len(ext.lab_values)} values identified")
    print(f"  Latency:  {result.processing_ms} ms")
    print("="*60 + "\n")


def save_result(result: PipelineResult, output_path: str = None):
    """Serializes the pipeline result to a JSON file."""
    if output_path is None:
        output_path = f"{Path(result.file_path).stem}_extracted.json"
    
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"Result exported to: {output_path}")


if __name__ == "__main__":
    main()
