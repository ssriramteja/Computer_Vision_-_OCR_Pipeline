import sys
import json
import logging
import tempfile
from pathlib import Path
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pipeline import OCRPipeline

st.set_page_config(
    page_title="Medical OCR Pipeline",
    layout="wide"
)

def main():
    st.title("Medical Document OCR Pipeline")
    st.subheader("Image Preprocessing | Region Detection | Tesseract OCR | Clinical NLP")

    @st.cache_resource
    def load_pipeline():
        logger.info("Loading OCR pipeline into Streamlit cache...")
        return OCRPipeline()

    pipeline = load_pipeline()

    # Sidebar: Document Upload and Config
    with st.sidebar:
        st.header("Document Input")
        uploaded_file = st.file_uploader(
            "Upload Image or PDF",
            type=["pdf", "png", "jpg", "jpeg"],
            label_visibility="visible"
        )
        st.divider()
        st.header("Pipeline Status")
        st.info("Status: Ready to process.")
        st.divider()
        st.markdown("### Supported Categories")
        st.markdown("- Prescriptions\n- Laboratory Reports\n- Discharge Summaries")

    # Main Application Logic
    if uploaded_file:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Executing pipeline operations..."):
                result = pipeline.process(tmp_path)
            
            logger.info(f"Successfully processed {uploaded_file.name}")
            
            # Layout: Results Display
            col_data, col_text = st.columns(2)

            with col_data:
                st.header("Extracted Data")
                ext = result.extracted

                # Metrics Section
                m1, m2, m3 = st.columns(3)
                m1.metric("Doc Type", result.doc_type)
                m2.metric("Latency", f"{result.processing_ms}ms")
                m3.metric("OCR Conf.", f"{result.avg_ocr_conf:.1f}%")

                st.divider()

                # Basic Information
                st.markdown("### Demographics & Source")
                st.markdown(f"**Patient Name:** {ext.patient_name or 'N/A'}")
                st.markdown(f"**Facility:** {ext.hospital or 'N/A'}")
                st.markdown(f"**Physician:** {ext.doctor_name  or 'N/A'}")
                st.markdown(f"**Date:** {ext.date or 'N/A'}")
                st.markdown(f"**MRN:** {ext.mrn or 'N/A'}")

                # Clinical Entities
                if ext.medications:
                    st.divider()
                    st.markdown("### Identified Medications")
                    for med in ext.medications:
                        st.markdown(f"- **{med['drug']}** ({med['dosage']})")

                if ext.lab_values:
                    st.divider()
                    st.markdown("### Laboratory Results")
                    for lab in ext.lab_values:
                        st.markdown(f"- **{lab['test']}**: {lab['value']} {lab['unit']}")

                if ext.diagnoses:
                    st.divider()
                    st.markdown("### Clinical Findings")
                    for dx in ext.diagnoses:
                        st.markdown(f"- {dx}")

            with col_text:
                st.header("OCR Output Pipeline")
                st.text_area(
                    "Aggregated Raw Text",
                    value=result.full_ocr_text,
                    height=400,
                    label_visibility="visible"
                )

                st.divider()
                st.header("Export JSON")
                st.json(result.to_dict())
                st.download_button(
                    label="Download Extraction Result",
                    data=json.dumps(result.to_dict(), indent=2),
                    file_name=f"{Path(uploaded_file.name).stem}_extracted.json",
                    mime="application/json"
                )

        except Exception as e:
            logger.error(f"UI Error processing {uploaded_file.name}: {str(e)}")
            st.error(f"Processing Failed: {str(e)}")
        finally:
            # Clean up temp file
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()
    else:
        st.info("Please upload a medical document via the sidebar to initiate parsing.")
        st.markdown("---")
        st.markdown("""
        ### Integration Guide
        1. **Pre-processing**: OpenCV adaptive thresholding and deskewing.
        2. **Region Detection**: Heuristic vertical segmentation (Fallback: YOLOv8).
        3. **Extraction**: Tesseract LSTM Engine followed by Clinical NER via spaCy.
        """)

if __name__ == "__main__":
    main()
