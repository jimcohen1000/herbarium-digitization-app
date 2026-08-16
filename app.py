import io
import os
import shutil
import tempfile
import zipfile
import google.generativeai as genai
import pandas as pd
from PIL import Image
from pyzbar.pyzbar import decode
import pytesseract
import streamlit as st
from streamlit_cropper import st_cropper

st.set_page_config(
    layout="wide", page_title="Herbarium Image-First Digitization"
)
st.title("Herbarium Image-First Databasing Tool")

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

# Sidebar: Institutional Settings
st.sidebar.header("1. Institutional Defaults")
inst_code = st.sidebar.text_input("institutionCode", value="WSU")
coll_code = st.sidebar.text_input("collectionCode", value="Herbarium")

# Sidebar: OCR Engine Choice
st.sidebar.header("2. Engine Settings")
ocr_engine = st.sidebar.radio(
    "Select OCR Engine", ["Gemini AI (Recommended)", "Tesseract (Local Crop)"]
)

api_key = ""
if ocr_engine == "Gemini AI (Recommended)":
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        value=st.secrets.get("GEMINI_API_KEY", ""),
    )


# Helper Functions
def run_gemini_ocr(img: Image.Image, key: str) -> str:
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Extract all text from the specimen label in this herbarium sheet image. "
            "Return only the verbatim label text."
        )
        response = model.generate_content([prompt, img])
        return response.text.strip()
    except Exception as e:
        return f"Gemini Error: {str(e)}"


def run_tesseract_ocr(cropped_img: Image.Image) -> str:
    raw_ocr = pytesseract.image_to_string(cropped_img)
    return " ".join(raw_ocr.split())


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

        # Session Progress & Navigation Toolbar
        nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([2, 1, 1, 1])
        with nav_col1:
            st.markdown(
                f"### Specimen {st.session_state.idx + 1} of {total_imgs}: `{os.path.basename(img_path)}`"
            )

        with nav_col2:
            if st.button("⬅️ Previous") and st.session_state.idx > 0:
                st.session_state.idx -= 1
                st.rerun()

        with nav_col3:
            if st.button("⏭️ Skip"):
                st.session_state.idx += 1
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
                st.rerun()

        st.divider()

        col1, col2 = st.columns([1, 1])

        # Left Column: Image Viewer / Cropper
        with col1:
            st.write("**Specimen View**")
            cropped_img = st_cropper(
                image, realtime_update=True, box_color="#00FF00"
            )

        # Right Column: Data Entry with Visual Side-by-Side Verification
        with col2:
            # 1. Barcode Detection & Visual Snippet
            st.markdown("#### 1. Barcode Identification")
            barcodes = decode(image)

            auto_barcode = ""
            barcode_crop = None

            if barcodes:
                b = barcodes[0]
                auto_barcode = b.data.decode("utf-8")
                # Auto-crop the detected barcode for visual confirmation
                rect = b.rect
                pad = 30
                w, h = image.size
                box = (
                    max(0, rect.left - pad),
                    max(0, rect.top - pad),
                    min(w, rect.left + rect.width + pad),
                    min(h, rect.top + rect.height + pad),
                )
                barcode_crop = image.crop(box)

            b_col1, b_col2 = st.columns([1, 1])
            with b_col1:
                if barcode_crop:
                    st.image(barcode_crop, caption="Auto-Detected Barcode")
                else:
                    st.caption("No barcode auto-detected. Use button below.")

            with b_col2:
                cat_num = st.text_input(
                    "Catalog Number (Barcode)", value=auto_barcode
                )
                if st.button("Read Barcode from Crop Box"):
                    crop_barcodes = decode(cropped_img)
                    if crop_barcodes:
                        cat_num = crop_barcodes[0].data.decode("utf-8")
                        st.success(f"Barcode found: {cat_num}")
                    else:
                        st.warning("No barcode detected in selected crop.")

            st.divider()

            # 2. Label OCR & Visual Snippet
            st.markdown("#### 2. Label OCR Verification")

            # Trigger OCR Execution
            if "ocr_text" not in st.session_state:
                st.session_state.ocr_text = ""

            if st.button(f"Run OCR ({ocr_engine.split()[0]})"):
                if ocr_engine == "Gemini AI (Recommended)":
                    if not api_key:
                        st.error(
                            "Please enter a Gemini API Key in the sidebar."
                        )
                    else:
                        with st.spinner("Gemini reading label..."):
                            st.session_state.ocr_text = run_gemini_ocr(
                                image, api_key
                            )
                else:
                    st.session_state.ocr_text = run_tesseract_ocr(cropped_img)

            o_col1, o_col2 = st.columns([1, 1])
            with o_col1:
                st.image(cropped_img, caption="Cropped Label Area")

            with o_col2:
                raw_label_text = st.text_area(
                    "Verbatim Label Text",
                    value=st.session_state.ocr_text,
                    height=200,
                )

            st.divider()

            # Save & Next Button
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
        st.success("Batch processing complete! Download your files on the left sidebar.")

# Export Section
st.sidebar.header("4. Export Session Data")
if st.session_state.records:
    df = pd.DataFrame(st.session_state.records)
    csv_bytes = df.to_csv(index=False).encode("utf-8")

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
