import io
import json
import os
import re
import pyzbar.pyzbar as pyzbar
import streamlit as st
import zxingcpp
from google import genai
from google.genai import types
from PIL import Image

# -----------------------------------------------------------------------------
# 1. Page Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Herbaria Specimen Digitizer", page_icon="🌿", layout="wide"
)

# Core Darwin Core terms mapped to digitization workflow
DWC_TERMS = [
    "catalogNumber",
    "scientificName",
    "recordedBy",
    "recordNumber",
    "eventDate",
    "verbatimEventDate",
    "country",
    "stateProvince",
    "county",
    "locality",
    "verbatimElevation",
    "decimalLatitude",
    "decimalLongitude",
    "habitat",
    "occurrenceRemarks",
    "verbatimLabel",
]

DEFAULT_DWC_RECORD = {term: "" for term in DWC_TERMS}

# Initialize form field session states if not present
for term in DWC_TERMS:
    if f"field_{term}" not in st.session_state:
        st.session_state[f"field_{term}"] = ""

if "dwc_record" not in st.session_state:
    st.session_state["dwc_record"] = DEFAULT_DWC_RECORD.copy()


def sync_record_to_widgets(record: dict):
    """Explicitly syncs a Darwin Core dict into Streamlit's text_input widget keys."""
    st.session_state["dwc_record"] = record
    for term in DWC_TERMS:
        val = record.get(term, "")
        st.session_state[f"field_{term}"] = str(val) if val is not None else ""


# -----------------------------------------------------------------------------
# 2. Local Barcode Detection Engine
# -----------------------------------------------------------------------------
def detect_barcodes(image: Image.Image) -> list:
    """Attempts local fast barcode extraction via zxingcpp and pyzbar."""
    found = []

    # Method 1: ZXing-CPP
    try:
        results = zxingcpp.read_barcodes(image)
        for res in results:
            if res.text and res.text not in found:
                found.append(res.text)
    except Exception:
        pass

    # Method 2: PyZBar fallback
    if not found:
        try:
            results = pyzbar.decode(image)
            for res in results:
                text = res.data.decode("utf-8")
                if text and text not in found:
                    found.append(text)
        except Exception:
            pass

    return found


# -----------------------------------------------------------------------------
# 3. Vision AI Label Extraction Engine
# -----------------------------------------------------------------------------
def run_gemini_parser(image: Image.Image, key: str) -> dict:
    """Sends specimen image to Gemini Vision AI for structured Darwin Core extraction."""
    # Downscale image copy for fast network transmission
    img_payload = image.copy()
    img_payload.thumbnail((1600, 1600), Image.Resampling.LANCZOS)

    client = genai.Client(api_key=key)

    # Valid model targets
    candidate_models = [
        "gemini-2.5-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
    ]

    system_instruction = """
    You are an expert botanical taxonomist and herbarium digitization specialist.
    Extract text from the botanical specimen label into JSON using these precise Darwin Core terms:
    
    - catalogNumber: Any barcode number or accession ID visible on the sheet.
    - scientificName: Full botanical name (Genus species authority).
    - recordedBy: Collector name(s).
    - recordNumber: Collector's field or collection number.
    - eventDate: ISO 8601 formatted collection date (YYYY-MM-DD).
    - verbatimEventDate: Date as typed directly on the label.
    - country: Country name.
    - stateProvince: State, province, or primary territory.
    - county: County, parish, or district.
    - locality: Specific description of location.
    - verbatimElevation: Altitude/elevation string as written.
    - decimalLatitude: Extracted or converted latitude in decimal degrees.
    - decimalLongitude: Extracted or converted longitude in decimal degrees.
    - habitat: Vegetation type, soil, substrate, or associated species.
    - occurrenceRemarks: Phenology, plant description, frequency, or notes.
    - verbatimLabel: Complete verbatim transcription of all text on the label.

    Output strictly valid JSON matching this schema. If a field is missing, return empty string "".
    """

    prompt = "Transcribe and extract all Darwin Core fields from this botanical specimen image."

    json_schema = {
        "type": "OBJECT",
        "properties": {term: {"type": "STRING"} for term in DWC_TERMS},
        "required": DWC_TERMS,
    }

    last_err = None
    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, img_payload],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=json_schema,
                    temperature=0.1,
                ),
            )
            raw_text = response.text
            parsed = json.loads(raw_text)

            clean_rec = DEFAULT_DWC_RECORD.copy()
            for k in DWC_TERMS:
                clean_rec[k] = parsed.get(k, "") or ""
            return clean_rec

        except Exception as e:
            last_err = e

    st.error(f"⚠️ Gemini API Execution Failed: {str(last_err)}")
    return DEFAULT_DWC_RECORD.copy()


# -----------------------------------------------------------------------------
# 4. Sidebar Controls & Input
# -----------------------------------------------------------------------------
st.sidebar.title("🌿 Options & Controls")

api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    value=os.getenv("GEMINI_API_KEY", ""),
    help="Get a free key from Google AI Studio.",
)

auto_parse = st.sidebar.checkbox(
    "⚡ Auto-parse image on load",
    value=True,
    help="Automatically run Vision AI whenever a new specimen image loads.",
)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader(
    "Upload Specimen Image", type=["jpg", "jpeg", "png", "tif", "tiff"]
)

is_sample = st.sidebar.button("📷 Load Sample Specimen")

# -----------------------------------------------------------------------------
# 5. Application Main Workflow
# -----------------------------------------------------------------------------
st.title("Herbarium Specimen Digitization Workbench")
st.markdown(
    "Automated barcode detection and multi-field label extraction powered by **Gemini Vision AI**."
)

image = None
img_identifier = None

if is_sample:
    sample_path = "sample.jpg"
    if os.path.exists(sample_path):
        image = Image.open(sample_path)
        img_identifier = "sample.jpg"
    else:
        st.warning(
            f"Sample image not found at `{sample_path}`. Upload a file manually."
        )
elif uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_identifier = uploaded_file.name

if image:
    col_img, col_form = st.columns([1, 1])

    with col_img:
        st.subheader("Specimen Preview")
        st.image(image, use_container_width=True)

        # Fast local barcode scanning
        barcodes = detect_barcodes(image)
        detected_catalog = barcodes[0] if barcodes else ""

        if detected_catalog:
            st.success(f"📌 **Detected Barcode:** `{detected_catalog}`")
        else:
            st.info("ℹ️ No barcode detected locally.")

    # Image load state tracking
    is_new_image = (
        st.session_state.get("current_img_name") != img_identifier
    )
    if is_new_image:
        st.session_state["current_img_name"] = img_identifier
        base_rec = DEFAULT_DWC_RECORD.copy()
        if detected_catalog:
            base_rec["catalogNumber"] = detected_catalog

        if auto_parse and api_key:
            with st.spinner("🤖 Vision AI reading specimen label..."):
                extracted = run_gemini_parser(image, api_key)
                if detected_catalog and not extracted.get("catalogNumber"):
                    extracted["catalogNumber"] = detected_catalog
                sync_record_to_widgets(extracted)
        else:
            sync_record_to_widgets(base_rec)

    with col_form:
        st.subheader("Darwin Core Record")

        btn_col, export_col = st.columns([1, 1])

        with btn_col:
            if st.button("🔄 Run / Re-Parse with Vision AI", type="primary"):
                if not api_key:
                    st.error(
                        "Please enter a valid Gemini API Key in the sidebar."
                    )
                else:
                    with st.spinner(
                        "🤖 Vision AI reading specimen label..."
                    ):
                        extracted = run_gemini_parser(image, api_key)
                        if detected_catalog and not extracted.get(
                            "catalogNumber"
                        ):
                            extracted["catalogNumber"] = detected_catalog
                        sync_record_to_widgets(extracted)
                        st.rerun()

        # Build Form UI
        t1, t2, t3, t4 = st.tabs(
            [
                "Taxonomy & Collector",
                "Locality & Date",
                "Remarks & Phenology",
                "Verbatim Text",
            ]
        )

        with t1:
            st.text_input("catalogNumber", key="field_catalogNumber")
            st.text_input("scientificName", key="field_scientificName")
            st.text_input("recordedBy", key="field_recordedBy")
            st.text_input("recordNumber", key="field_recordNumber")

        with t2:
            st.text_input("eventDate", key="field_eventDate")
            st.text_input("verbatimEventDate", key="field_verbatimEventDate")
            st.text_input("country", key="field_country")
            st.text_input("stateProvince", key="field_stateProvince")
            st.text_input("county", key="field_county")
            st.text_input("locality", key="field_locality")
            st.text_input("verbatimElevation", key="field_verbatimElevation")
            c1, c2 = st.columns(2)
            with c1:
                st.text_input("decimalLatitude", key="field_decimalLatitude")
            with c2:
                st.text_input("decimalLongitude", key="field_decimalLongitude")

        with t3:
            st.text_area("habitat", key="field_habitat")
            st.text_area("occurrenceRemarks", key="field_occurrenceRemarks")

        with t4:
            st.text_area("verbatimLabel", key="field_verbatimLabel", height=200)

        # Collect current widget input values
        current_record = {
            term: st.session_state.get(f"field_{term}", "")
            for term in DWC_TERMS
        }

        st.markdown("---")
        st.download_button(
            label="💾 Export Darwin Core JSON",
            data=json.dumps(current_record, indent=2),
            file_name=f"DwC_{current_record.get('catalogNumber') or 'specimen'}.json",
            mime="application/json",
        )
else:
    st.info("👈 Upload an image or click **Load Sample Specimen** to start.")
