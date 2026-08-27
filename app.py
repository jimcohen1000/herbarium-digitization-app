import csv
import gc
import io
import json
import time
from PIL import Image
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

# ------------------------------------------------------------------------------
# 1. Page Configuration & Defaults
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Herbarium Specimen Digitizer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DWC_RECORD = {
    "catalogNumber": "",
    "barcodeBox": [],
    "labelBox": [],
    "scientificName": "",
    "genus": "",
    "specificEpithet": "",
    "scientificNameAuthorship": "",
    "identifiedBy": "",
    "recordedBy": "",
    "associatedCollectors": "",
    "recordNumber": "",
    "eventDate": "",
    "verbatimEventDate": "",
    "year": "",
    "month": "",
    "day": "",
    "occurrenceRemarks": "",
    "habitat": "",
    "substrate": "",
    "associatedTaxa": "",
    "reproductiveCondition": "",
    "country": "",
    "stateProvince": "",
    "county": "",
    "municipality": "",
    "locality": "",
    "geocodingSearchTerm": "",
    "locationRemarks": "",
    "decimalLatitude": "",
    "decimalLongitude": "",
    "verbatimCoordinates": "",
    "elevationNumber": "",
    "minimumElevationInMeters": "",
    "maximumElevationInMeters": "",
    "verbatimElevation": "",
    "verbatimLabel": "",
    "_inputTokens": "0",
    "_outputTokens": "0",
    "_totalTokens": "0",
}

# ------------------------------------------------------------------------------
# 2. Session State Setup
# ------------------------------------------------------------------------------
if "last_error" not in st.session_state:
    st.session_state.last_error = None
if "parsed_data" not in st.session_state:
    st.session_state.parsed_data = None
if "last_filename" not in st.session_state:
    st.session_state.last_filename = None

# ------------------------------------------------------------------------------
# 3. Resilient Gemini Vision Parser
# ------------------------------------------------------------------------------
def run_gemini_parser(img: Image.Image, key: str) -> dict:
    if not key:
        res = DEFAULT_DWC_RECORD.copy()
        res["verbatimLabel"] = "Error: Missing Gemini API Key."
        st.session_state.last_error = "Gemini API key is missing. Please enter your key in the sidebar or secrets."
        return res

    img_payload = None
    try:
        # Create thumbnail copy to prevent RAM bloat and high-resolution timeouts
        img_payload = img.copy()
        img_payload.thumbnail((1600, 1600), Image.Resampling.LANCZOS)

        client = genai.Client(api_key=key)

        candidate_models = [
            "gemini-3.6-flash",
            "gemini-3.7-flash",
            "gemini-3.5-flash",
        ]

        prompt = """
        Examine this herbarium specimen sheet. Locate the primary specimen label, plant specimen, and barcode sticker.
        Extract data into standard Symbiota / Darwin Core fields.

        STRICT STANDARDIZATION RULES:
        1. COUNTRY: If the specimen is from the US, set "country" to EXACTLY "United States".
        2. TAXONOMY: Extract "scientificName" (e.g. Pinus ponderosa Douglas ex C.Lawson). Break out "genus" and "specificEpithet".
        3. DATES: Extract "eventDate" (e.g. 1984-06-15). Break out "year", "month", and "day". Extract "verbatimEventDate" as written.
        4. COORDINATES: Extract "verbatimCoordinates" as printed. Convert any DMS/UTM into decimal degrees as "decimalLatitude" and "decimalLongitude". Ensure West and South values are negative numbers.
           IMPORTANT: If no numerical coordinates are printed on the label, leave BOTH "decimalLatitude" AND "decimalLongitude" as empty strings ("").
        5. ELEVATION: Convert numeric values to meters. 
           - If a single elevation value is present (not a range), populate "elevationNumber" with that single value, and leave BOTH "minimumElevationInMeters" AND "maximumElevationInMeters" as empty strings ("").
           - If an elevation range is present (e.g., 2000-2500 m), leave "elevationNumber" as an empty string (""), set "minimumElevationInMeters" to the lower value, and "maximumElevationInMeters" to the upper value.
        6. PHENOLOGY: Assign "reproductiveCondition" to EXACTLY one of: ["In Flower", "In Fruit", "Flowering and Fruiting", "Flower Buds", "Vegetative", "Sterile", "Cones", "Spores"] or empty string.
        7. GEOCODING SEARCH TERM: Extract a clean, simple landmark or named place for geocoding (e.g. "Lee Valley Reservoir" or "Greer"). Omit distances/directions.

        Return ONLY a JSON object:
        {
            "catalogNumber": "Extracted barcode or catalog ID number",
            "barcodeBox": [ymin, xmin, ymax, xmax],
            "labelBox": [ymin, xmin, ymax, xmax],
            "scientificName": "Full taxon name including author if present",
            "genus": "Genus name",
            "specificEpithet": "Species epithet",
            "scientificNameAuthorship": "Author name string",
            "identifiedBy": "Determiner name",
            "recordedBy": "Collector name(s)",
            "associatedCollectors": "Co-collectors list",
            "recordNumber": "Collector field number",
            "eventDate": "Collection date in YYYY-MM-DD format if possible",
            "verbatimEventDate": "Raw date string as written",
            "year": "4-digit year",
            "month": "Numeric month (1-12)",
            "day": "Numeric day (1-31)",
            "occurrenceRemarks": "Plant description or specimen observations",
            "habitat": "Habitat or community notes",
            "substrate": "Soil or growing medium notes",
            "associatedTaxa": "Associated species list",
            "reproductiveCondition": "Exact match from phenology terms or empty string",
            "country": "Country name",
            "stateProvince": "State or Province",
            "county": "County or Parish",
            "municipality": "City, town, or municipality",
            "locality": "Detailed locality description",
            "geocodingSearchTerm": "Simplified landmark name for geocoding",
            "locationRemarks": "Additional location notes",
            "decimalLatitude": "Numeric latitude in decimal degrees or empty string",
            "decimalLongitude": "Numeric longitude in decimal degrees or empty string",
            "verbatimCoordinates": "Raw coordinate string from label",
            "elevationNumber": "Single elevation value in meters or empty string if range",
            "minimumElevationInMeters": "Lower elevation in meters if range, else empty string",
            "maximumElevationInMeters": "Upper elevation in meters if range, else empty string",
            "verbatimElevation": "Raw elevation string as recorded on label",
            "verbatimLabel": "Full exact verbatim label text"
        }
        """

        last_error = None
        for model_name in candidate_models:
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[prompt, img_payload],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json"
                        ),
                    )
                    if response and response.text:
                        parsed = json.loads(response.text)
                        merged = DEFAULT_DWC_RECORD.copy()
                        merged.update(parsed)

                        if hasattr(response, "usage_metadata") and response.usage_metadata:
                            merged["_inputTokens"] = str(response.usage_metadata.prompt_token_count)
                            merged["_outputTokens"] = str(response.usage_metadata.candidates_token_count)
                            merged["_totalTokens"] = str(response.usage_metadata.total_token_count)

                        st.session_state.last_error = None
                        return merged

                except Exception as err:
                    last_error = err
                    err_str = str(err)
                    # Retry transient 503 unavailable or 429 rate limit errors with exponential sleep
                    if any(code in err_str for code in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"]):
                        time.sleep((2 ** attempt) + 0.5)
                        continue
                    # Bypass deprecated endpoints immediately
                    elif any(code in err_str for code in ["404", "NOT_FOUND"]):
                        break
                    else:
                        break

        if last_error:
            st.session_state.last_error = f"Gemini API Call Failed: {str(last_error)}"
            raise last_error

    except Exception as e:
        res = DEFAULT_DWC_RECORD.copy()
        res["verbatimLabel"] = f"API Error: {str(e)}"
        st.session_state.last_error = f"Gemini Execution Error: {str(e)}"
        return res

    finally:
        # EXPLICIT RAM CLEANUP: Replaces leaking buffers to stop OOM app restarts
        if img_payload is not None:
            try:
                img_payload.close()
            except Exception:
                pass
            del img_payload
        gc.collect()

def crop_bounding_box(image: Image.Image, box: list) -> Image.Image:
    """Utility to crop label images using normalized bounding boxes."""
    if not box or len(box) != 4:
        return None
    w, h = image.size
    ymin, xmin, ymax, xmax = box
    if all(isinstance(v, float) and v <= 1.0 for v in box):
        left, top, right, bottom = xmin * w, ymin * h, xmax * w, ymax * h
    else:
        left, top, right, bottom = (xmin / 1000) * w, (ymin / 1000) * h, (xmax / 1000) * w, (ymax / 1000) * h
    return image.crop((left, top, right, bottom))

# ------------------------------------------------------------------------------
# 4. Streamlit UI Layout
# ------------------------------------------------------------------------------
st.title("🌿 Herbarium Darwin Core Digitizer")

# Sidebar Configuration
with st.sidebar:
    st.header("🔑 Credentials & Settings")
    api_key = st.text_input("Gemini API Key", type="password")
    if not api_key and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.caption("✓ Using API Key from Streamlit Secrets")

    st.markdown("---")
    st.markdown("**Session RAM Management**")
    if st.button("🧹 Clear Memory & Cache"):
        st.session_state.parsed_data = None
        st.session_state.last_filename = None
        st.session_state.last_error = None
        gc.collect()
        st.success("RAM flushed successfully.")

# Main Workspace
uploaded_file = st.file_uploader("Upload Specimen Image Sheet", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file is not None:
    # Memory safety: Clear old image payload buffers when opening a new specimen file
    if st.session_state.last_filename != uploaded_file.name:
        st.session_state.parsed_data = None
        st.session_state.last_error = None
        st.session_state.last_filename = uploaded_file.name
        gc.collect()

    img = Image.open(uploaded_file)
    
    col_img, col_data = st.columns([1, 1])

    # Left Column: Specimen Viewer & Action Controls
    with col_img:
        st.subheader("Specimen Sheet")
        st.image(img, width="stretch")

        if st.button("🚀 Run / Re-Parse Specimen", type="primary"):
            with st.spinner("Processing specimen sheet with Gemini Vision..."):
                st.session_state.parsed_data = run_gemini_parser(img, api_key)

        # Label Crop Preview (if bounding box returned)
        if st.session_state.parsed_data and st.session_state.parsed_data.get("labelBox"):
            st.markdown("**Detected Specimen Label Crop**")
            label_crop = crop_bounding_box(img, st.session_state.parsed_data["labelBox"])
            if label_crop:
                st.image(label_crop, width="content")
                label_crop.close()

    # Right Column: Extracted Data Editor & CSV Export
    with col_data:
        st.subheader("Darwin Core Extraction")
        
        if st.session_state.last_error:
            st.error(st.session_state.last_error)

        if st.session_state.parsed_data:
            rec = st.session_state.parsed_data

            # Token Usage Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Input Tokens", rec.get("_inputTokens", "0"))
            m2.metric("Output Tokens", rec.get("_outputTokens", "0"))
            m3.metric("Total Tokens", rec.get("_totalTokens", "0"))

            st.markdown("---")

            # Editable Form Fields grouped by Darwin Core Categories
            with st.form("dwc_edit_form"):
                st.markdown("#### 1. Catalog & Identification")
                rec["catalogNumber"] = st.text_input("catalogNumber", value=rec.get("catalogNumber", ""))
                rec["scientificName"] = st.text_input("scientificName", value=rec.get("scientificName", ""))
                
                c1, c2 = st.columns(2)
                rec["genus"] = c1.text_input("genus", value=rec.get("genus", ""))
                rec["specificEpithet"] = c2.text_input("specificEpithet", value=rec.get("specificEpithet", ""))
                
                rec["scientificNameAuthorship"] = st.text_input("scientificNameAuthorship", value=rec.get("scientificNameAuthorship", ""))
                rec["identifiedBy"] = st.text_input("identifiedBy", value=rec.get("identifiedBy", ""))

                st.markdown("#### 2. Collection Event")
                c3, c4 = st.columns(2)
                rec["recordedBy"] = c3.text_input("recordedBy", value=rec.get("recordedBy", ""))
                rec["recordNumber"] = c4.text_input("recordNumber", value=rec.get("recordNumber", ""))
                
                rec["eventDate"] = st.text_input("eventDate (YYYY-MM-DD)", value=rec.get("eventDate", ""))
                rec["verbatimEventDate"] = st.text_input("verbatimEventDate", value=rec.get("verbatimEventDate", ""))

                st.markdown("#### 3. Locality & Geography")
                rec["country"] = st.text_input("country", value=rec.get("country", ""))
                c5, c6 = st.columns(2)
                rec["stateProvince"] = c5.text_input("stateProvince", value=rec.get("stateProvince", ""))
                rec["county"] = c6.text_input("county", value=rec.get("county", ""))
                
                rec["locality"] = st.text_area("locality", value=rec.get("locality", ""))
                rec["geocodingSearchTerm"] = st.text_input("geocodingSearchTerm", value=rec.get("geocodingSearchTerm", ""))

                c7, c8 = st.columns(2)
                rec["decimalLatitude"] = c7.text_input("decimalLatitude", value=rec.get("decimalLatitude", ""))
                rec["decimalLongitude"] = c8.text_input("decimalLongitude", value=rec.get("decimalLongitude", ""))
                rec["verbatimCoordinates"] = st.text_input("verbatimCoordinates", value=rec.get("verbatimCoordinates", ""))

                st.markdown("#### 4. Environment & Verbatim Label")
                rec["habitat"] = st.text_area("habitat", value=rec.get("habitat", ""))
                rec["reproductiveCondition"] = st.text_input("reproductiveCondition", value=rec.get("reproductiveCondition", ""))
                rec["verbatimLabel"] = st.text_area("verbatimLabel", value=rec.get("verbatimLabel", ""), height=120)

                update_submitted = st.form_submit_button("Save Edits")
                if update_submitted:
                    st.session_state.parsed_data = rec
                    st.success("Form fields updated.")

            # CSV / JSON Export options
            st.markdown("### 📥 Download Results")
            df = pd.DataFrame([st.session_state.parsed_data])
            csv_data = df.to_csv(index=False).encode('utf-8')
            json_data = json.dumps(st.session_state.parsed_data, indent=2).encode('utf-8')

            exp_c1, exp_c2 = st.columns(2)
            exp_c1.download_button(
                label="Download Darwin Core CSV",
                data=csv_data,
                file_name=f"{rec.get('catalogNumber', 'dwc_record')}.csv",
                mime="text/csv",
            )
            exp_c2.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"{rec.get('catalogNumber', 'dwc_record')}.json",
                mime="application/json",
            )

    img.close()

# End of Streamlit execution cycle garbage collector call
gc.collect()
