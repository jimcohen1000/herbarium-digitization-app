import io
import json
import os
import re
import shutil
import tempfile
import zipfile
import cv2
import easyocr
import google.generativeai as genai
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from pyzbar.pyzbar import decode
import pytesseract
import streamlit as st
from streamlit_cropper import st_cropper
import zxingcpp

st.set_page_config(
    layout="wide", page_title="Herbarium Image-First Digitization"
)
st.title("Herbarium Image-First Databasing Tool")


# Cache EasyOCR model in memory
@st.cache_resource
def load_easyocr_reader():
    return easyocr.Reader(["en"], gpu=False)


reader = load_easyocr_reader()

# Initialize Session State
if "records" not in st.session_state:
    st.session_state.records = []
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "image_paths" not in st.session_state:
    st.session_state.image_paths = []
if "work_dir" not in st.session_state:
    st.session_state.work_dir = tempfile.mkdtemp()
if "out_dir" not in st.session_state:
    st.session_state.out_dir = tempfile.mkdtemp()

# Parsed Field State Buffer
default_fields = {
    "scientificName": "",
    "recordedBy": "",
    "recordNumber": "",
    "eventDate": "",
    "country": "USA",
    "stateProvince": "",
    "county": "",
    "locality": "",
    "habitat": "",
    "verbatimLabel": "",
}

if "parsed_fields" not in st.session_state:
    st.session_state.parsed_fields = default_fields.copy()

# Sidebar: Institutional Settings
st.sidebar.header("1. Institutional Defaults")
inst_code = st.sidebar.text_input("institutionCode", value="WSU")
coll_code = st.sidebar.text_input("collectionCode", value="Herbarium")

# Sidebar: OCR Engine Choice
st.sidebar.header("2. Engine Settings")
ocr_engine = st.sidebar.radio(
    "Select OCR Engine",
    [
        "EasyOCR (Deep Learning - Crop)",
        "Tesseract (Local Crop - Free)",
        "Gemini AI (Full Sheet Auto-Parse)",
    ],
)

api_key = ""
if ocr_engine == "Gemini AI (Full Sheet Auto-Parse)":
    api_key = st.sidebar.text_input(
        "Gemini API Key (Personal Account)",
        type="password",
        value=st.secrets.get("GEMINI_API_KEY", ""),
    )


# Helper Functions
def decode_barcode_fullres(img: Image.Image) -> str:
    """Decodes barcodes using OpenCV adaptive binarization, quiet zone padding, and ZXing across 4 rotations."""
    img_padded = ImageOps.expand(img, border=40, fill="white")

    # Binarize with OpenCV for low contrast / faint barcodes
    open_cv_image = np.array(img_padded.convert("L"))
    binarized = cv2.adaptiveThreshold(
        open_cv_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    pil_binarized = Image.fromarray(binarized)

    for angle in [0, 90, 180, 270]:
        rot_raw = (
            img_padded if angle == 0 else img_padded.rotate(angle, expand=True)
        )
        rot_bin = (
            pil_binarized
            if angle == 0
            else pil_binarized.rotate(angle, expand=True)
        )

        for target in [rot_raw, rot_bin]:
            try:
                results = zxingcpp.read_barcodes(target)
                if results and results[0].text:
                    return results[0].text
            except Exception:
                pass

            try:
                barcodes = decode(target)
                if barcodes:
                    return barcodes[0].data.decode("utf-8")
            except Exception:
                pass

    return ""


def parse_label_heuristics(text: str) -> dict:
    """Extracts common Symbiota fields from text using pattern matching."""
    parsed = default_fields.copy()
    parsed["verbatimLabel"] = text

    if not text:
        return parsed

    # Scientific Name (Latin Binomial heuristic: Genus species)
    sci_match = re.search(
        r"\b([A-Z][a-z]{2,}\s+[a-z]{2,}(?:\s+(?:var\.|subsp\.|f\.)\s+[a-z]{2,})?)\b",
        text,
    )
    if sci_match:
        parsed["scientificName"] = sci_match.group(1)

    # Date
    date_match = re.search(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b",
        text,
        re.IGNORECASE,
    )
    if date_match:
        parsed["eventDate"] = date_match.group(0)

    # Collector
    coll_match = re.search(
        r"(?:Coll(?:ector)?s?|Leg)\.?:?\s*([A-Z][a-zA-Z\.\s,]+?)(?=\s+\d|\n|$)",
        text,
        re.IGNORECASE,
    )
    if coll_match:
        parsed["recordedBy"] = coll_match.group(1).strip()

    # Collector Number
    num_match = re.search(
        r"(?:No\.|#|Num\.?)\s*(\d+[A-Za-z]?)", text, re.IGNORECASE
    )
    if num_match:
        parsed["recordNumber"] = num_match.group(1)

    # County
    county_match = re.search(
        r"([A-Z][a-zA-Z\s]+?)\s+Co(?:unty)?\b", text, re.IGNORECASE
    )
    if county_match:
        parsed["county"] = county_match.group(1).strip() + " County"

    return parsed


def run_gemini_parser(img: Image.Image, key: str) -> dict:
    """Uses Gemini API to extract and parse Symbiota fields as structured JSON."""
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = """
        Extract all text from the specimen label in this image and parse it into Symbiota / Darwin Core fields.
        Return ONLY a JSON object matching this schema:
        {
            "scientificName": "Genus species infraspecies",
            "recordedBy": "Collector name(s)",
            "recordNumber": "Collector number",
            "eventDate": "Collection date",
            "country": "Country",
            "stateProvince": "State or Province",
            "county": "County",
            "locality": "Detailed locality description",
            "habitat": "Habitat or substrate notes",
            "verbatimLabel": "Exact verbatim text on label"
        }
        """
        response = model.generate_content(
            [prompt, img],
            generation_config={"response_mime_type": "application/json"},
        )
        return json.loads(response.text)
    except Exception as e:
        res = default_fields.copy()
        res["verbatimLabel"] = f"Gemini Error: {str(e)}"
        return res


# Sidebar: Upload Batch
st.sidebar.header("3. Upload Batch")
uploaded_files = st.sidebar.file_uploader(
    "Upload ZIP archive or image files (JPG, PNG, TIF)",
    type=["zip", "jpg", "jpeg", "png", "tif", "tiff"],
    accept_multiple_files=True,
)

if uploaded_files and not st.session_state.image_paths:
    for uploaded_file in uploaded_files:
        if uploaded_file.name.lower().endswith(".zip"):
            with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                zip_ref.extractall(st.session_state.work_dir)
        else:
            file_path = os.path.join(
                st.session_state.work_dir, uploaded_file.name
            )
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

    valid_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(st.session_state.work_dir)
        for f in files
        if f.lower().endswith(valid_exts)
    ]
    st.session_state.image_paths = sorted(paths)
    st.rerun()

# Main Interactive Interface
if st.session_state.image_paths:
    total_imgs = len(st.session_state.image_paths)

    if st.session_state.idx < total_imgs:
        img_path = st.session_state.image_paths[st.session_state.idx]
        image = Image.open(img_path)

        # Navigation Toolbar
        nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([2, 1, 1, 1])
        with nav_col1:
            st.markdown(
                f"### Specimen {st.session_state.idx + 1} of {total_imgs}: `{os.path.basename(img_path)}`"
            )
        with nav_col2:
            if st.button("⬅️ Previous") and st.session_state.idx > 0:
                st.session_state.idx -= 1
                st.session_state.parsed_fields = default_fields.copy()
                st.rerun()
        with nav_col3:
            if st.button("⏭️ Skip"):
                st.session_state.idx += 1
                st.session_state.parsed_fields = default_fields.copy()
                st.rerun()
        with nav_col4:
            if st.button("↩️ Undo Last Save") and st.session_state.records:
                last_record = st.session_state.records.pop()
                last_file = os.path.join(
                    st.session_state.out_dir, last_record["associatedMedia"]
                )
                if os.path.exists(last_file):
                    os.remove(last_file)
                st.session_state.idx = max(0, st.session_state.idx - 1)
                st.session_state.parsed_fields = default_fields.copy()
                st.rerun()

        st.divider()

        # Left Column: Dual Cropper & Zoom
        col1, col2 = st.columns([1, 1])

        with col1:
            zoom_factor = st.slider(
                "🔍 Image Zoom Level",
                min_value=1.0,
                max_value=3.0,
                value=1.0,
                step=0.25,
            )

            if zoom_factor > 1.0:
                w, h = image.size
                image_display = image.resize(
                    (int(w * zoom_factor), int(h * zoom_factor))
                )
            else:
                image_display = image

            tab_label, tab_barcode = st.tabs(
                ["1. Crop Label (OCR)", "2. Crop Barcode"]
            )

            with tab_label:
                st.caption(
                    "Position green box over specimen label text for OCR."
                )
                label_crop = st_cropper(
                    image_display,
                    realtime_update=True,
                    box_color="#00FF00",
                    key="label_cropper",
                )

            with tab_barcode:
                st.caption(
                    "Position blue box over barcode if auto-detection fails."
                )
                barcode_crop = st_cropper(
                    image_display,
                    realtime_update=True,
                    box_color="#0000FF",
                    key="barcode_cropper",
                )

        # Right Column: Barcode & Parsed Symbiota Fields
        with col2:
            st.markdown("#### 1. Barcode Identification")

            auto_barcode = decode_barcode_fullres(image)
            cat_num = st.text_input(
                "Catalog Number (Barcode)", value=auto_barcode
            )

            b_col1, b_col2 = st.columns([1, 1])
            with b_col1:
                if st.button("Read Barcode from Crop Box"):
                    cropped_code = decode_barcode_fullres(barcode_crop)
                    if cropped_code:
                        cat_num = cropped_code
                        st.success(f"Barcode Detected: {cat_num}")
                    else:
                        st.error(
                            "Could not read barcode from crop. Enter manually above."
                        )
            with b_col2:
                st.image(
                    barcode_crop, caption="Barcode Crop Preview", width=200
                )

            st.divider()

            st.markdown("#### 2. Label Parsing & Symbiota Fields")

            engine_label = ocr_engine.split()[0]
            if st.button(f"Run OCR & Parse ({engine_label})"):
                if ocr_engine == "EasyOCR (Deep Learning - Crop)":
                    with st.spinner("EasyOCR reading label..."):
                        img_np = np.array(label_crop)
                        results = reader.readtext(img_np, detail=0)
                        raw_text = " ".join(results)
                        st.session_state.parsed_fields = (
                            parse_label_heuristics(raw_text)
                        )
                elif ocr_engine == "Tesseract (Local Crop - Free)":
                    raw_text = pytesseract.image_to_string(label_crop)
                    st.session_state.parsed_fields = parse_label_heuristics(
                        " ".join(raw_text.split())
                    )
                else:
                    if not api_key:
                        st.error(
                            "Please enter a Gemini API Key in the sidebar."
                        )
                    else:
                        with st.spinner("Gemini parsing label fields..."):
                            st.session_state.parsed_fields = run_gemini_parser(
                                image, api_key
                            )

            pf = st.session_state.parsed_fields

            # Interactive Symbiota Input Fields
            sci_name = st.text_input(
                "scientificName", value=pf.get("scientificName", "")
            )

            f_col1, f_col2, f_col3 = st.columns([1, 1, 1])
            with f_col1:
                rec_by = st.text_input(
                    "recordedBy (Collector)", value=pf.get("recordedBy", "")
                )
            with f_col2:
                rec_num = st.text_input(
                    "recordNumber", value=pf.get("recordNumber", "")
                )
            with f_col3:
                ev_date = st.text_input(
                    "eventDate", value=pf.get("eventDate", "")
                )

            geo_col1, geo_col2, geo_col3 = st.columns([1, 1, 1])
            with geo_col1:
                cntry = st.text_input("country", value=pf.get("country", "USA"))
            with geo_col2:
                state_prov = st.text_input(
                    "stateProvince", value=pf.get("stateProvince", "")
                )
            with geo_col3:
                county = st.text_input("county", value=pf.get("county", ""))

            locality = st.text_input("locality", value=pf.get("locality", ""))
            habitat = st.text_input("habitat", value=pf.get("habitat", ""))
            verb_label = st.text_area(
                "verbatimLabel", value=pf.get("verbatimLabel", ""), height=120
            )

            st.divider()

            if st.button("💾 Save Record & Next Specimen", type="primary"):
                if not cat_num:
                    st.error("Catalog Number is required.")
                else:
                    ext = os.path.splitext(img_path)[1]
                    new_filename = f"{cat_num}{ext}"
                    dest_path = os.path.join(
                        st.session_state.out_dir, new_filename
                    )

                    shutil.copy(img_path, dest_path)

                    st.session_state.records.append(
                        {
                            "institutionCode": inst_code,
                            "collectionCode": coll_code,
                            "catalogNumber": cat_num,
                            "scientificName": sci_name,
                            "recordedBy": rec_by,
                            "recordNumber": rec_num,
                            "eventDate": ev_date,
                            "country": cntry,
                            "stateProvince": state_prov,
                            "county": county,
                            "locality": locality,
                            "habitat": habitat,
                            "verbatimLabel": verb_label,
                            "associatedMedia": new_filename,
                        }
                    )

                    st.session_state.parsed_fields = default_fields.copy()
                    st.session_state.idx += 1
                    st.rerun()
    else:
        st.success("Batch processing complete!")

# Live Symbiota Import Spreadsheet Table
st.divider()
st.markdown("### 📊 Live Symbiota Import Spreadsheet")

if st.session_state.records:
    st.caption(
        "You can edit any cell directly in the table below before downloading."
    )
    df = pd.DataFrame(st.session_state.records)

    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key="spreadsheet_editor",
    )

    st.session_state.records = edited_df.to_dict("records")
else:
    st.info("No records saved yet in this session.")

# Sidebar Export Controls
st.sidebar.header("4. Export Session Data")
if st.session_state.records:
    export_df = pd.DataFrame(st.session_state.records)
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")

    st.sidebar.download_button(
        label="📥 Download Symbiota CSV",
        data=csv_bytes,
        file_name="symbiota_import.csv",
        mime="text/csv",
    )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for fname in os.listdir(st.session_state.out_dir):
            fpath = os.path.join(st.session_state.out_dir, fname)
            zf.write(fpath, fname)

    st.sidebar.download_button(
        label="📥 Download Renamed Images (.zip)",
        data=zip_buffer.getvalue(),
        file_name="renamed_specimens.zip",
        mime="application/zip",
    )
