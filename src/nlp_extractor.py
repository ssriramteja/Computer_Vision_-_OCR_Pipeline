import re
import spacy
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ExtractedDocument:
    """Represents a structured medical document extracted from OCR text.

    Attributes:
        doc_type (str): Classified type of the document (e.g., PRESCRIPTION).
        patient_name (Optional[str]): Identified patient name.
        date (Optional[str]): Relevant date (e.g., service date, discharge date).
        doctor_name (Optional[str]): Identified physician name.
        hospital (Optional[str]): Identified healthcare facility.
        mrn (Optional[str]): Medical Record Number.
        dob (Optional[str]): Date of Birth.
        medications (List[Dict[str, str]]): List of identified medications and dosages.
        lab_values (List[Dict[str, str]]): List of identified lab results.
        diagnoses (List[str]): List of identified conditions or diagnoses.
        raw_text (str): The source OCR text.
    """
    doc_type: str
    patient_name: Optional[str] = None
    date: Optional[str] = None
    doctor_name: Optional[str] = None
    hospital: Optional[str] = None
    mrn: Optional[str] = None
    dob: Optional[str] = None
    medications: List[Dict[str, str]] = field(default_factory=list)
    lab_values: List[Dict[str, str]] = field(default_factory=list)
    diagnoses: List[str] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Converts the extracted data into a JSON-serializable dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the document.
        """
        return {
            "doc_type": self.doc_type,
            "patient_name": self.patient_name,
            "date": self.date,
            "doctor_name": self.doctor_name,
            "hospital": self.hospital,
            "mrn": self.mrn,
            "dob": self.dob,
            "medications": self.medications,
            "lab_values": self.lab_values,
            "diagnoses": self.diagnoses,
        }


class MedicalNLPExtractor:
    """Extracts clinical entities and structured data from medical OCR text.

    Utilizes spaCy for Named Entity Recognition (NER) and pre-defined regex
    patterns for domain-specific medical fields.
    """

    # Domain Keywords
    DRUG_KEYWORDS = {
        "metformin", "lisinopril", "atorvastatin", "aspirin",
        "metoprolol", "azithromycin", "prednisone", "albuterol",
        "ceftriaxone", "insulin", "lasix", "furosemide",
        "amoxicillin", "ciprofloxacin", "hydroxychloroquine",
        "nimodipine", "heparin", "warfarin", "clopidogrel"
    }

    DIAGNOSIS_KEYWORDS = [
        "diagnosis", "assessment", "impression", "findings",
        "condition", "disease", "disorder", "syndrome",
        "pneumonia", "diabetes", "hypertension", "stroke",
        "failure", "infection", "fracture", "carcinoma"
    ]

    # Regular Expression Patterns
    LAB_PATTERN = re.compile(
        r"(Glucose|BUN|Creatinine|Sodium|Potassium|HbA1c|Cholesterol|HDL|LDL|"
        r"Triglycerides|WBC|RBC|Hemoglobin|Hematocrit|Platelets|TSH|Troponin|BNP)"
        r"\s+(\d+\.?\d*)\s*(mg/dL|%|mEq/L|pg/mL|ng/mL|U/L|mmol/L)?",
        re.IGNORECASE
    )

    DATE_PATTERN = re.compile(
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
    )
    MRN_PATTERN = re.compile(r"\b(MRN[:\s]*[\w-]+|\b\d{7,10}\b)", re.IGNORECASE)
    DOSAGE_PATTERN = re.compile(r"\b(\d+\.?\d*\s*(?:mg|mcg|ml|units?|tabs?))\b", re.IGNORECASE)

    def __init__(self, model: str = "en_core_web_sm"):
        """Initializes the NLP extractor with a specific spaCy model.

        Args:
            model: The name of the spaCy model to load.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        try:
            self.nlp = spacy.load(model)
            logger.info(f"Loaded spaCy model: {model}")
        except OSError:
            logger.error(f"spaCy model '{model}' not found.")
            raise RuntimeError(f"Model not found. Run: python -m spacy download {model}")

    def _detect_doc_type(self, text: str) -> str:
        """Determines the document classification based on text content.

        Args:
            text: Source OCR text.

        Returns:
            str: Document type classification.
        """
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["prescription", "rx"]):
            return "PRESCRIPTION"
        elif any(kw in text_lower for kw in ["laboratory", "lab report"]):
            return "LAB_REPORT"
        elif "discharge" in text_lower:
            return "DISCHARGE_SUMMARY"
        return "GENERAL_MEDICAL"

    def _extract_medications(self, text: str) -> List[Dict[str, str]]:
        """Identifies medications and associated dosages from text.

        Args:
            text: Source OCR text.

        Returns:
            List[Dict[str, str]]: List of found drugs and dosages.
        """
        meds = []
        tokens = text.lower().split()
        for i, tok in enumerate(tokens):
            clean_tok = re.sub(r"[^a-z]", "", tok)
            if clean_tok in self.DRUG_KEYWORDS:
                # Scoped context for dosage detection
                context = " ".join(tokens[i:i+4])
                dosage_match = self.DOSAGE_PATTERN.search(context)
                meds.append({
                    "drug": clean_tok.capitalize(),
                    "dosage": dosage_match.group(0) if dosage_match else "N/A"
                })
        return meds

    def _extract_lab_values(self, text: str) -> List[Dict[str, str]]:
        """Parses laboratory test results including units and values.

        Args:
            text: Source OCR text.

        Returns:
            List[Dict[str, str]]: List of lab result mappings.
        """
        labs = []
        for match in self.LAB_PATTERN.finditer(text):
            labs.append({
                "test": match.group(1),
                "value": match.group(2),
                "unit": match.group(3) or ""
            })
        return labs

    def _extract_diagnoses(self, text: str) -> List[str]:
        """Identifies diagnostic statements or findings from the text.

        Args:
            text: Source OCR text.

        Returns:
            List[str]: Best-effort identified diagnoses strings.
        """
        diagnoses = []
        lines = text.split("\n")
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in self.DIAGNOSIS_KEYWORDS):
                clean_line = line.strip()
                if 10 < len(clean_line) < 200:
                    diagnoses.append(clean_line)
        return list(dict.fromkeys(diagnoses))[:5]  # De-duplicate and limit

    def extract(self, text: str) -> ExtractedDocument:
        """Performs a full extraction sequence on the OCR text.

        Args:
            text: Source OCR text.

        Returns:
            ExtractedDocument: The populated structured document object.
        """
        doc = self.nlp(text)
        doc_type = self._detect_doc_type(text)

        result = ExtractedDocument(
            doc_type=doc_type,
            raw_text=text
        )

        # Entity Recognition mapping
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                if result.patient_name is None:
                    result.patient_name = ent.text
                elif result.doctor_name is None and any(kw in ent.text.lower() for kw in ["dr.", "dr ", "doctor"]):
                    result.doctor_name = ent.text
            elif ent.label_ == "ORG" and result.hospital is None:
                result.hospital = ent.text
            elif ent.label_ == "DATE" and result.date is None:
                if not re.search(r"\d{4}", ent.text) or len(ent.text) > 20: continue
                result.date = ent.text

        # High-precision Regex overrides
        dates = self.DATE_PATTERN.findall(text)
        if dates and result.date is None:
            result.date = dates[0]

        mrn_match = self.MRN_PATTERN.search(text)
        if mrn_match:
            result.mrn = mrn_match.group(0)

        # Specialty Extraction
        result.medications = self._extract_medications(text)
        result.lab_values = self._extract_lab_values(text)
        result.diagnoses = self._extract_diagnoses(text)

        # Physician fallback detection
        if result.doctor_name is None:
            dr_match = re.search(r"Dr\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+", text)
            if dr_match:
                result.doctor_name = dr_match.group(0)

        return result
