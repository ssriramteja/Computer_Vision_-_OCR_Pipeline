# Computer Vision + Medical OCR Pipeline

A professional-grade pipeline for analyzing, OCR-ing, and extracting structured clinical data from medical documents (Prescriptions, Lab Reports, Discharge Summaries).

## Key Features
- **Advanced Preprocessing**: OpenCV-based grayscale conversion, denoising, adaptive thresholding, deskewing, and upscaling.
- **Logical Segmentation**: Heuristic vertical segmentation and YOLOv8 support for document structure analysis.
- **Tesseract Engine**: Optimized Tesseract configuration using LSTM for high-accuracy text extraction.
- **Clinical NLP**: Medical Entity Recognition using spaCy and precision regex patterns for drugs, lab values, and diagnoses.
- **Streamlit Dashboard**: A clean, professional web interface for document upload and analysis visualization.

## Project Structure
- `src/`: Core processing logic (preprocessor, detector, ocr_engine, nlp_extractor, pipeline).
- `ui/`: Streamlit web application.
- `data/`: Synthetic document generation and sample text inputs.

## Installation

### 1. System Dependencies
Install Tesseract and Poppler (for PDF support):
```bash
brew install tesseract poppler
```

### 2. Python Environment
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Usage

### Web Interface
```bash
streamlit run ui/app.py
```

### Command Line
```bash
python src/pipeline.py path/to/document.pdf
```

## License
MIT
