"""
Lab Report OCR Module
─────────────────────
Extracts numeric lab values from an uploaded report image.

Strategy (in order):
1. pytesseract + Tesseract binary  (best accuracy, needs Tesseract installed)
2. Pure regex on OpenCV pre-processed image text via pytesseract if available
3. Graceful fallback: returns empty dict so the frontend shows manual entry

The endpoint always returns a dict of recognised fields; missing fields are simply
absent so the frontend can show placeholders.
"""

import re
import logging
import importlib.util
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Optional tesseract ────────────────────────────────────────────────────────
PYTESSERACT_AVAILABLE = importlib.util.find_spec("pytesseract") is not None
TESSERACT_OK = False

if PYTESSERACT_AVAILABLE:
    import pytesseract
    # Common Windows install paths — set explicitly so PATH doesn't matter
    _WIN_PATHS = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    import os as _os
    for _p in _WIN_PATHS:
        if _os.path.isfile(_p):
            pytesseract.pytesseract.tesseract_cmd = _p
            break
    try:
        pytesseract.get_tesseract_version()
        TESSERACT_OK = True
        logger.info("Tesseract OCR available ✓ (%s)", pytesseract.pytesseract.tesseract_cmd)
    except Exception as _e:
        logger.warning("pytesseract installed but Tesseract binary not found – OCR disabled. (%s)", _e)


# ── Field patterns ────────────────────────────────────────────────────────────
# Each entry:  (field_key, [list of label regexes], unit_hint, scale_factor)
# scale_factor converts the raw number to the unit used in SymptomData

FIELD_PATTERNS = [
    # bilirubin  mg/dL
    ("bilirubin",
     [r"bilirubin\s*(?:total|t\.?bili)?",
      r"t\.\s*bilirubin", r"total\s*bili"],
     "mg/dl", 1.0),

    # albumin  g/dL
    ("albumin",
     [r"albumin"],
     "g/dl", 1.0),

    # cholesterol  mg/dL
    ("cholesterol",
     [r"cholesterol\s*(?:total|t\.?chol)?",
      r"total\s*cholesterol", r"chol\b"],
     "mg/dl", 1.0),

    # AST  U/L
    ("ast",
     [r"ast\b", r"aspartate\s*(?:amino)?trans?",
      r"sgot"],
     "u/l", 1.0),

    # Alkaline Phosphatase  U/L
    ("alk_phos",
     [r"alk(?:aline)?\s*phos(?:phatase)?",
      r"alp\b", r"a\.p\b"],
     "u/l", 1.0),

    # Copper  µg/dL
    ("copper",
     [r"copper\b", r"cu\b"],
     "ug/dl", 1.0),

    # Triglycerides  mg/dL
    ("triglycerides",
     [r"triglyceride", r"trig\b", r"vldl\s*chol"],
     "mg/dl", 1.0),

    # Platelets  ×10³/µL  — stored as raw number (e.g. 220 → 220000 if in K)
    ("platelets",
     [r"platelet\s*(?:count)?", r"plt\b", r"thrombocyte"],
     "k/ul", 1000.0),   # most reports show in K/µL

    # Prothrombin Time  seconds
    ("prothrombin",
     [r"prothrombin\s*(?:time)?", r"pt\b(?!\s*inr)", r"p\.t\b"],
     "sec", 1.0),

    # ALT – not in SymptomData but useful to pass through
    ("alt",
     [r"alt\b", r"alanine\s*(?:amino)?trans?", r"sgpt"],
     "u/l", 1.0),
]


# ── Image pre-processing ──────────────────────────────────────────────────────

def _preprocess(image: np.ndarray) -> np.ndarray:
    """Convert to high-contrast greyscale for better OCR."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Resize if too small
    h, w = gray.shape
    if w < 1200:
        scale = 1200 / w
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)

    # Denoise + threshold
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _extract_text(image: np.ndarray) -> str:
    """Run OCR and return full text string."""
    if not TESSERACT_OK:
        return ""

    processed = _preprocess(image)
    pil_img = Image.fromarray(processed)
    config = "--psm 6 --oem 3"          # Assume uniform block of text
    text = pytesseract.image_to_string(pil_img, config=config)
    return text


# ── Value extraction ──────────────────────────────────────────────────────────

def _parse_number(token: str) -> Optional[float]:
    """Extract first float/int from a token string."""
    m = re.search(r"(\d{1,6}(?:[.,]\d{1,3})?)", token)
    if m:
        return float(m.group(1).replace(",", "."))
    return None


def _search_field(text: str, label_patterns: list) -> Optional[float]:
    """
    Search `text` for any label pattern and return the first numeric value
    found within 120 characters after the label.
    """
    text_l = text.lower()
    for pat in label_patterns:
        for m in re.finditer(pat, text_l):
            snippet = text_l[m.end(): m.end() + 120]
            val = _parse_number(snippet)
            if val is not None:
                return val
    return None


def extract_lab_values(image_bytes: bytes) -> Dict[str, float]:
    """
    Main entry point.

    Parameters
    ----------
    image_bytes : raw bytes of the uploaded image file

    Returns
    -------
    dict mapping field names (matching SymptomData) to float values.
    Empty dict if OCR is unavailable or no values found.
    """
    if not TESSERACT_OK:
        logger.info("OCR not available – returning empty result.")
        return {"_ocr_available": False}

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"_ocr_available": True, "_error": "Could not decode image"}

        text = _extract_text(img)
        logger.debug("OCR raw text:\n%s", text[:500])

        results: Dict[str, float] = {"_ocr_available": True}

        for field_key, patterns, _unit, scale in FIELD_PATTERNS:
            raw = _search_field(text, patterns)
            if raw is not None:
                # Apply scale (e.g. platelets in K → raw units)
                value = raw * scale
                # Sanity clip to avoid absurd outliers
                value = _sanity_clip(field_key, value)
                if value is not None:
                    results[field_key] = round(value, 2)

        logger.info("OCR extracted %d lab fields", len(results) - 1)
        return results

    except Exception as e:
        logger.error("OCR extraction failed: %s", e)
        return {"_ocr_available": True, "_error": str(e)}


# ── Sanity bounds (match SymptomData constraints) ─────────────────────────────

_BOUNDS = {
    "bilirubin":     (0.1,  20.0),
    "albumin":       (1.0,   5.0),
    "cholesterol":   (100,  400),
    "ast":           (10,  1000),
    "alk_phos":      (40,   500),
    "copper":        (10,   200),
    "triglycerides": (50,   500),
    "platelets":     (50000, 500000),
    "prothrombin":   (8,    20),
    "alt":           (5,   1000),
}


def _sanity_clip(field: str, value: float) -> Optional[float]:
    bounds = _BOUNDS.get(field)
    if bounds is None:
        return value
    lo, hi = bounds
    if lo <= value <= hi:
        return value
    # Try without the platelet scale if it overshoots
    if field == "platelets" and value > hi:
        smaller = value / 1000.0      # Maybe already in units not K
        if lo <= smaller <= hi:
            return smaller
    return None  # discard implausible value
