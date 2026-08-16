import io
import os
import shutil
import tempfile
import zipfile
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


# Cache EasyOCR model in memory so it loads once at startup
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
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""

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
        "Gemini AI (Full Sheet Auto-Detect)",
    ],
)

api_key = ""
if ocr_engine == "Gemini AI (Full Sheet Auto-Detect)":
    api_key = st.sidebar.text_input(
        "Gemini API Key (Personal Account)",
        type="password",
        value=st.secrets.get("GEMINI_API_KEY", ""),
    )


# Helper Functions
def decode_barcode_advanced(img: Image.Image) -> str:
    """Multi-angle, dual-engine (zxing-cpp + pyzbar) barcode reader with padding."""
    # Add white margin (Quiet Zone) required for barcode decoding
    img_padded = ImageOps.expand(img, border=30, fill="white")
    gray = img_padded.convert("L")
    contrast = ImageEnhance.Contrast(gray).enhance(2.5)

    # Test across 4 rotations (0°, 90°, 180°, 270°)
    for angle in [0, 90, 180, 270]:
        rotated_img = (
            img_padded if angle == 0 else img_padded.rotate(angle, expand=True)
        )
        rotated_contrast = (
            contrast if angle == 0 else contrast.rotate(angle, expand=True)
        )

        # Primary Engine: ZXing-CPP
        try:
            results = zxingcpp.read_barcodes(rotated_img)
            if results and results[0].text:
                return results[0].text
        except Exception:
            pass

        # Fallback Engine: PyZBar
        try:
            barcodes = decode(rotated_contrast)
            if barcodes:
                return barcodes[0].data.decode("utf-8")
        except Exception:
            pass

    return ""


def run_easyocr(cropped_img: Image.Image) -> str:
    img_np = np.array(cropped_img)
    results = reader.readtext(img_np, detail=0)
    return " ".join(results)


def run_tesseract_ocr(cropped_img: Image.Image) -> str:
    raw_ocr = pytesseract.image_to_string(cropped_img)
    return " ".join(raw_ocr.split())


def run_gemini_ocr(img: Image.Image, key: str) -> str:
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = "Extract all text from the specimen label in this herbarium sheet image. Return only verbatim label text."
        response = model.generate_content([prompt, img])
        return response.text.strip()
    except Exception as e:
        return f"Gemini Error: {str(e)}"


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

        # Toolbar
        nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([2, 1, 1, 1])
        with nav_col1:
            st.markdown(
                f"### Specimen {st.session_state.idx + 1} of {total_imgs}: `{os.path.basename(img_path)}`"
            )
        with nav_col2:
            if st.button("⬅️ Previous") and st.session_state.idx > 0:
                st.session_state.idx -= 1
                st.session_state.ocr_text = ""
                st.rerun()
        with nav_col3:
            if st.button("⏭️ Skip"):
                st.session_state.idx += 1
                st.session_state.ocr_text = ""
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
                st.session_state.ocr_text = ""
                st.rerun()

        st.divider()

        # Left Column: Dual Cropper & Zoom Controls
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

        # Right Column: Barcode & OCR Fields
        with col2:
            st.markdown("#### 1. Barcode Identification")

            # Try multi-angle full-image auto decode first
            auto_barcode = decode_barcode_advanced(image)

            cat_num = st.text_input(
                "Catalog Number (Barcode)", value=auto_barcode
            )

            b_col1, b_col2 = st.columns([1, 1])
            with b_col1:
                if st.button("Read Barcode from Crop Box"):
                    cropped_code = decode_barcode_advanced(barcode_crop)
                    if cropped_code:
                        cat_num = cropped_code
                        st.success(f"Barcode Detected: {cat_num}")
                    else:
                        st.error(
                            "Could not read barcode from selected crop. Please enter manually above."
                        )
            with b_col2:
                st.image(
                    barcode_crop, caption="Barcode Crop Preview", width=200
                )

            st.divider()

            st.markdown("#### 2. Label OCR Verification")

            engine_label = ocr_engine.split()[0]
            if st.button(f"Run OCR ({engine_label})"):
                if ocr_engine == "EasyOCR (Deep Learning - Crop)":
                    with st.spinner("EasyOCR processing crop..."):
                        st.session_state.ocr_text = run_easyocr(label_crop)
                elif ocr_engine == "Tesseract (Local Crop - Free)":
                    st.session_state.ocr_text = run_tesseract_ocr(label_crop)
                else:
                    if not api_key:
                        st.error(
                            "Please enter a Gemini API Key in the sidebar."
                        )
                    else:
                        with st.spinner("Gemini reading full sheet..."):
                            st.session_state.ocr_text = run_gemini_ocr(
                                image, api_key
                            )

            o_col1, o_col2 = st.columns([1, 1])
            with o_col1:
                st.image(label_crop, caption="Label Crop Preview")
            with o_col2:
                raw_label_text = st.text_area(
                    "Verbatim Label Text",
                    value=st.session_state.ocr_text,
                    height=180,
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
                            "associatedMedia": new_filename,
                            "verbatimLabel": raw_label_text,
                        }
                    )

                    st.session_state.ocr_text = ""
                    st.session_state.idx += 1
                    st.rerun()
    else:
        st.success("Batch processing complete!")

# Interactive Editable Table
st.divider()
st.markdown("### 📊 Live Symbiota Import Spreadsheet")

if st.session_state.records:
    st.caption(
        "You can edit cells directly in the table below before downloading."
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
