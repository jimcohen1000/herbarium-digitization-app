import io
import json
import os
import shutil
import tempfile
import zipfile
import cv2
from google import genai
from google.genai import types
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from pyzbar.pyzbar import decode
import streamlit as st
from streamlit_cropper import st_cropper
import zxingcpp

st.set_page_config(
    layout="wide", page_title="Herbarium Image-First Digitization"
)
st.title("Herbarium Digitization Tool (Free Vision AI)")

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

default_fields = {
    "catalogNumber": "",
    "barcodeBox": [],
    "labelBox": [],
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

# Sidebar Settings
st.sidebar.header("1. Institutional Defaults")
inst_code = st.sidebar.text_input("institutionCode", value="WSU")
coll_code = st.sidebar.text_input("collectionCode", value="Herbarium")

st.sidebar.header("2. Free API Key")
api_key = st.sidebar.text_input(
    "Gemini API Key (Free from AI Studio)",
    type="password",
    value=st.secrets.get("GEMINI_API_KEY", ""),
    help="Default key is loaded from secrets if available. Clear and type to override.",
)


# Crop helper function for 0-1000 scale bounding boxes
def crop_box_1000(img: Image.Image, box: list) -> Image.Image:
    """Crops an image given a [ymin, xmin, ymax, xmax] box normalized to 0-1000 scale."""
    if not box or len(box) != 4:
        return None
    try:
        w, h = img.size
        ymin, xmin, ymax, xmax = box
        abs_left = max(0, int((xmin / 1000.0) * w))
        abs_top = max(0, int((ymin / 1000.0) * h))
        abs_right = min(w, int((xmax / 1000.0) * w))
        abs_bottom = min(h, int((ymax / 1000.0) * h))
        if abs_right > abs_left and abs_bottom > abs_top:
            return img.crop((abs_left, abs_top, abs_right, abs_bottom))
    except Exception:
        pass
    return None


# Barcode Decoder: Crop directly from full-resolution source pixels
def decode_barcode_fullres(img: Image.Image, crop_box: dict = None) -> str:
    """Slices original full-res pixels using relative crop coordinates and applies adaptive binarization."""
    target_img = img

    if crop_box and crop_box.get("width", 0) > 0:
        left = int(crop_box["left"])
        top = int(crop_box["top"])
        right = int(crop_box["left"] + crop_box["width"])
        bottom = int(crop_box["top"] + crop_box["height"])

        target_img = img.crop((left, top, right, bottom))

    padded = ImageOps.expand(target_img, border=40, fill="white")

    cv_img = np.array(padded.convert("L"))
    binarized = cv2.adaptiveThreshold(
        cv_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    pil_bin = Image.fromarray(binarized)

    for angle in [0, 90, 180, 270]:
        rot_raw = padded if angle == 0 else padded.rotate(angle, expand=True)
        rot_bin = pil_bin if angle == 0 else pil_bin.rotate(angle, expand=True)

        for candidate in [rot_raw, rot_bin]:
            try:
                results = zxingcpp.read_barcodes(candidate)
                if results and results[0].text:
                    return results[0].text
            except Exception:
                pass

            try:
                barcodes = decode(candidate)
                if barcodes:
                    return barcodes[0].data.decode("utf-8")
            except Exception:
                pass

    return ""


# Vision API Label Parser with Bounding Box Detection
def run_gemini_parser(img: Image.Image, key: str) -> dict:
    """Uses Google GenAI SDK to structure label and bounding box data into Symbiota JSON fields."""
    try:
        client = genai.Client(api_key=key)

        candidate_models = [
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-2.5-flash",
        ]

        prompt = """
        Examine this herbarium specimen sheet. Locate the primary specimen label and any barcode or catalog stickers.
        Extract the data into Symbiota/Darwin Core fields.
        Also locate bounding boxes normalized to a 0-1000 scale for the barcode sticker and for the primary specimen text label in format [ymin, xmin, ymax, xmax].

        Return ONLY a JSON object matching this schema:
        {
            "catalogNumber": "Extracted barcode or catalog ID number",
            "barcodeBox": [ymin, xmin, ymax, xmax],
            "labelBox": [ymin, xmin, ymax, xmax],
            "scientificName": "Full genus species infraspecies",
            "recordedBy": "Collector name(s)",
            "recordNumber": "Collector number",
            "eventDate": "Collection date",
            "country": "Country",
            "stateProvince": "State or Province",
            "county": "County",
            "locality": "Detailed locality description",
            "habitat": "Habitat or substrate notes",
            "verbatimLabel": "Full exact verbatim label text"
        }
        """

        last_error = None
        for model_name in candidate_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, img],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    ),
                )
                if response and response.text:
                    return json.loads(response.text)
            except Exception as err:
                last_error = err
                continue

        if last_error:
            raise last_error

    except Exception as e:
        res = default_fields.copy()
        res["verbatimLabel"] = f"API Error: {str(e)}"
        return res


# Sidebar: Upload Batch
st.sidebar.header("3. Upload Batch")
uploaded_files = st.sidebar.file_uploader(
    "Upload ZIP archive or images (JPG, PNG, TIF)",
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

# Main Digitization Workspace
if st.session_state.image_paths:
    total_imgs = len(st.session_state.image_paths)

    if st.session_state.idx < total_imgs:
        img_path = st.session_state.image_paths[st.session_state.idx]
        image = Image.open(img_path)

        # Toolbar
        nav1, nav2, nav3, nav4 = st.columns([2, 1, 1, 1])
        with nav1:
            st.markdown(
                f"### Specimen {st.session_state.idx + 1} of {total_imgs}: `{os.path.basename(img_path)}`"
            )
        with nav2:
            if st.button("⬅️ Previous") and st.session_state.idx > 0:
                st.session_state.idx -= 1
                st.session_state.parsed_fields = default_fields.copy()
                st.rerun()
        with nav3:
            if st.button("⏭️ Skip"):
                st.session_state.idx += 1
                st.session_state.parsed_fields = default_fields.copy()
                st.rerun()
        with nav4:
            if st.button("↩️ Undo Last") and st.session_state.records:
                last_rec = st.session_state.records.pop()
                last_f = os.path.join(
                    st.session_state.out_dir, last_rec["associatedMedia"]
                )
                if os.path.exists(last_f):
                    os.remove(last_f)
                st.session_state.idx = max(0, st.session_state.idx - 1)
                st.session_state.parsed_fields = default_fields.copy()
                st.rerun()

        st.divider()

        col1, col2 = st.columns([1, 1])

        # Left Column: Image & Manual Barcode Crop Box
        with col1:
            st.caption(
                "Full Specimen View. Draw blue box over barcode if auto-detect fails."
            )
            barcode_box = st_cropper(
                image,
                realtime_update=True,
                box_color="#0000FF",
                key="barcode_cropper",
                return_type="box",
            )

        # Right Column: AI Zoomed Crops & Data Fields
        with col2:
            pf = st.session_state.parsed_fields

            # Show Zoomed Crops when detected
            b_crop = crop_box_1000(image, pf.get("barcodeBox"))
            l_crop = crop_box_1000(image, pf.get("labelBox"))

            if b_crop or l_crop:
                st.markdown("#### 🔎 AI Zoomed Detection Views")
                zcol1, zcol2 = st.columns(2)
                with zcol1:
                    if b_crop:
                        st.image(
                            b_crop,
                            caption="Detected Barcode Region",
                            use_container_width=True,
                        )
                with zcol2:
                    if l_crop:
                        st.image(
                            l_crop,
                            caption="Detected Label Region",
                            use_container_width=True,
                        )
                st.divider()

            st.markdown("#### 1. Barcode Identification")

            # Fallback Logic: Local Decoder -> AI Catalog Number
            auto_code = decode_barcode_fullres(image)
            if not auto_code and pf.get("catalogNumber"):
                auto_code = pf.get("catalogNumber")

            cat_num = st.text_input("Catalog Number (Barcode)", value=auto_code)

            if st.button("Read Barcode from Blue Crop Box"):
                cropped_code = decode_barcode_fullres(
                    image, crop_box=barcode_box
                )
                if cropped_code:
                    cat_num = cropped_code
                    st.success(f"Detected: {cat_num}")
                else:
                    st.error("No barcode detected in crop box. Enter manually.")

            st.divider()

            st.markdown("#### 2. Vision AI Label Parsing")

            if st.button("🤖 Parse Sheet with Free Vision AI", type="primary"):
                if not api_key:
                    st.error(
                        "Please enter a free Gemini API Key in the sidebar."
                    )
                else:
                    with st.spinner("Analyzing sheet with Gemini Vision AI..."):
                        st.session_state.parsed_fields = run_gemini_parser(
                            image, api_key
                        )
                        st.rerun()

            # Form Input Boxes
            sci_name = st.text_input(
                "scientificName", value=pf.get("scientificName", "")
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                rec_by = st.text_input(
                    "recordedBy", value=pf.get("recordedBy", "")
                )
            with c2:
                rec_num = st.text_input(
                    "recordNumber", value=pf.get("recordNumber", "")
                )
            with c3:
                ev_date = st.text_input(
                    "eventDate", value=pf.get("eventDate", "")
                )

            g1, g2, g3 = st.columns(3)
            with g1:
                cntry = st.text_input("country", value=pf.get("country", "USA"))
            with g2:
                state_prov = st.text_input(
                    "stateProvince", value=pf.get("stateProvince", "")
                )
            with g3:
                county = st.text_input("county", value=pf.get("county", ""))

            locality = st.text_input("locality", value=pf.get("locality", ""))
            habitat = st.text_input("habitat", value=pf.get("habitat", ""))
            verb_label = st.text_area(
                "verbatimLabel", value=pf.get("verbatimLabel", ""), height=100
            )

            st.divider()

            if st.button("💾 Save Record & Next Specimen"):
                if not cat_num:
                    st.error("Catalog Number is required before saving.")
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

# Live Symbiota Export Spreadsheet & Image Review
st.divider()
st.markdown("### 📊 Live Symbiota Import Spreadsheet")

if st.session_state.records:
    df = pd.DataFrame(st.session_state.records)
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key="spreadsheet_editor",
    )

    # Sync user spreadsheet edits back to session state
    st.session_state.records = edited_df.to_dict("records")

    st.markdown("### 🖼️ Saved Specimen Record Viewer")
    rec_labels = [
        f"Row {i+1}: [{r.get('catalogNumber', 'No-ID')}] {r.get('scientificName', 'Unidentified')}"
        for i, r in enumerate(st.session_state.records)
    ]
    selected_idx = st.selectbox(
        "Select a saved record to inspect its image:",
        options=range(len(rec_labels)),
        format_func=lambda x: rec_labels[x],
    )

    if selected_idx < len(st.session_state.records):
        rec = st.session_state.records[selected_idx]
        img_name = rec.get("associatedMedia", "")
        saved_img_path = os.path.join(st.session_state.out_dir, img_name)

        rcol1, rcol2 = st.columns([1, 1])
        with rcol1:
            if os.path.exists(saved_img_path):
                st.image(
                    saved_img_path,
                    caption=f"Renamed File: {img_name}",
                    use_container_width=True,
                )
            else:
                st.warning("Saved image file not found.")
        with rcol2:
            st.markdown("**Saved Record Details**")
            st.json(rec)
else:
    st.info("No records saved in current session.")

# Sidebar Downloads
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
