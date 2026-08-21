import io
import json
import os
import re
import shutil
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
st.title("Herbarium Digitization Tool (Weber State WSCO)")

# Persistent Local Work Directories
WORK_DIR = "./batch_input"
OUT_DIR = "./batch_output"
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Initialize Session State
if "records" not in st.session_state:
    st.session_state.records = []
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "page_data" not in st.session_state:
    st.session_state.page_data = (
        {}
    )  # Caches parsed/edited form data & barcodes per image
if "image_paths" not in st.session_state:
    st.session_state.image_paths = []

default_fields = {
    "catalogNumber": "",
    "barcodeBox": [],
    "labelBox": [],
    "scientificName": "",
    "genus": "",
    "specificEpithet": "",
    "recordedBy": "",
    "recordNumber": "",
    "eventDate": "",
    "year": "",
    "month": "",
    "day": "",
    "occurrenceRemarks": "",
    "habitat": "",
    "substrate": "",
    "associatedTaxa": "",
    "reproductiveCondition": "",
    "country": "United States",
    "stateProvince": "",
    "county": "",
    "municipality": "",
    "locality": "",
    "locationRemarks": "",
    "decimalLatitude": "",
    "decimalLongitude": "",
    "verbatimCoordinates": "",
    "minimumElevationInMeters": "",
    "maximumElevationInMeters": "",
    "verbatimElevation": "",
    "verbatimLabel": "",
}

# Sidebar Settings
st.sidebar.header("1. Institutional Defaults")
inst_code = st.sidebar.text_input("institutionCode", value="Weber State")
coll_code = st.sidebar.text_input("collectionCode", value="WSCO")

st.sidebar.header("2. Free API Key & Processing")
api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    value=st.secrets.get("GEMINI_API_KEY", ""),
    help="Default key is loaded from secrets if available. Clear and type to override.",
)

auto_parse = st.sidebar.checkbox(
    "⚡ Auto-parse image on load",
    value=True,
    help="Uncheck to browse images manually without calling Vision AI automatically.",
)


# Safe Zip Extraction Function (Zip-Slip Security Fix)
def safe_extract_zip(zip_file, target_dir):
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        for member in zip_ref.infolist():
            target_path = os.path.abspath(
                os.path.join(target_dir, member.filename)
            )
            if not target_path.startswith(os.path.abspath(target_dir)):
                raise Exception(
                    "Security Error: Attempted Path Traversal in Zip File"
                )
            zip_ref.extract(member, target_dir)


# Crop helper function for 0-1000 scale bounding boxes
def crop_box_1000(img: Image.Image, box: list) -> Image.Image:
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


# Barcode Decoder
def decode_barcode_fullres(img: Image.Image, crop_box: dict = None) -> str:
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


# Vision API Label Parser
def run_gemini_parser(img: Image.Image, key: str) -> dict:
    try:
        client = genai.Client(api_key=key)
        candidate_models = [
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-2.5-flash",
        ]

        prompt = """
        Examine this herbarium specimen sheet. Locate the primary specimen label, plant specimen, and barcode sticker.
        Extract data into standard Symbiota / Darwin Core fields.

        STRICT STANDARDIZATION RULES:
        1. COUNTRY: If the specimen is from the US, set "country" to EXACTLY "United States".
        2. TAXONOMY: Extract "scientificName". Break out "genus" and "specificEpithet".
        3. DATES: Extract "eventDate" (YYYY-MM-DD). Break out integer values for "year", "month", and "day".
        4. COORDINATES: Convert DMS/UTM into decimal degrees as "decimalLatitude" and "decimalLongitude". West/South must be negative numbers.
        5. ELEVATION: Convert values to meters. If single value, set BOTH "minimumElevationInMeters" AND "maximumElevationInMeters" equal.
        6. PHENOLOGY: Assign "reproductiveCondition" to EXACTLY one of: ["In Flower", "In Fruit", "Flowering and Fruiting", "Flower Buds", "Vegetative", "Sterile", "Cones", "Spores"]. Leave blank if unclear.

        Return ONLY a JSON object matching this schema:
        {
            "catalogNumber": "Extracted barcode or catalog ID number",
            "barcodeBox": [ymin, xmin, ymax, xmax],
            "labelBox": [ymin, xmin, ymax, xmax],
            "scientificName": "", "genus": "", "specificEpithet": "",
            "recordedBy": "", "recordNumber": "", "eventDate": "",
            "year": "", "month": "", "day": "", "occurrenceRemarks": "",
            "habitat": "", "substrate": "", "associatedTaxa": "",
            "reproductiveCondition": "", "country": "United States",
            "stateProvince": "", "county": "", "municipality": "",
            "locality": "", "locationRemarks": "", "decimalLatitude": "",
            "decimalLongitude": "", "verbatimCoordinates": "",
            "minimumElevationInMeters": "", "maximumElevationInMeters": "",
            "verbatimElevation": "", "verbatimLabel": ""
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
        res["verbatimLabel"] = f"API Parsing Error occurred: {str(e)}"
        return res


# Sidebar Upload Batch
st.sidebar.header("3. Upload Batch")
uploaded_files = st.sidebar.file_uploader(
    "Upload ZIP archive or images",
    type=["zip", "jpg", "jpeg", "png", "tif", "tiff"],
    accept_multiple_files=True,
)

if uploaded_files and not st.session_state.image_paths:
    for uploaded_file in uploaded_files:
        if uploaded_file.name.lower().endswith(".zip"):
            safe_extract_zip(uploaded_file, WORK_DIR)
        else:
            file_path = os.path.join(WORK_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

    valid_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(WORK_DIR)
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
        current_idx = st.session_state.idx

        # CACHED INITIALIZATION: Run barcode scan & Vision AI ONCE per image index
        if current_idx not in st.session_state.page_data:
            detected_barcode = decode_barcode_fullres(image)

            if auto_parse and api_key:
                with st.spinner(
                    f"🤖 Auto-parsing specimen {current_idx + 1} with Vision AI..."
                ):
                    parsed = run_gemini_parser(image, api_key)
                    if not parsed.get("catalogNumber") and detected_barcode:
                        parsed["catalogNumber"] = detected_barcode
                    st.session_state.page_data[current_idx] = parsed
            else:
                data = default_fields.copy()
                data["catalogNumber"] = detected_barcode
                st.session_state.page_data[current_idx] = data

        pf = st.session_state.page_data[current_idx]

        # Toolbar
        nav1, nav2, nav3, nav4 = st.columns([2, 1, 1, 1])
        with nav1:
            st.markdown(
                f"### Specimen {current_idx + 1} of {total_imgs}: `{os.path.basename(img_path)}`"
            )
        with nav2:
            if st.button("⬅️ Previous") and current_idx > 0:
                st.session_state.idx -= 1
                st.rerun()
        with nav3:
            if st.button("⏭️ Skip") and current_idx < total_imgs - 1:
                st.session_state.idx += 1
                st.rerun()
        with nav4:
            if st.button("↩️ Undo Last") and st.session_state.records:
                last_rec = st.session_state.records.pop()
                last_f = os.path.join(OUT_DIR, last_rec["associatedMedia"])
                if os.path.exists(last_f):
                    os.remove(last_f)
                st.session_state.idx = max(0, st.session_state.idx - 1)
                st.rerun()

        st.divider()

        col1, col2 = st.columns([1, 1])

        # Left Column: Image & Cropper
        with col1:
            st.caption("Full Specimen View. Draw box over barcode if needed.")
            barcode_box = st_cropper(
                image,
                realtime_update=True,
                box_color="#0000FF",
                key=f"cropper_{current_idx}",
                return_type="box",
            )

        # Right Column: Data Entry Form
        with col2:
            b_crop = crop_box_1000(image, pf.get("barcodeBox"))
            l_crop = crop_box_1000(image, pf.get("labelBox"))

            if b_crop or l_crop:
                st.markdown("#### 🔎 AI Zoomed Detection Views")
                zcol1, zcol2 = st.columns(2)
                with zcol1:
                    if b_crop:
                        st.image(
                            b_crop,
                            caption="Detected Barcode",
                            use_container_width=True,
                        )
                with zcol2:
                    if l_crop:
                        st.image(
                            l_crop,
                            caption="Detected Label",
                            use_container_width=True,
                        )
                st.divider()

            st.markdown("#### 1. Barcode Identification")
            cat_num = st.text_input(
                "Catalog Number (Barcode)", value=pf.get("catalogNumber", "")
            )

            if st.button("Read Barcode from Manual Crop Box"):
                cropped_code = decode_barcode_fullres(
                    image, crop_box=barcode_box
                )
                if cropped_code:
                    cat_num = cropped_code
                    pf["catalogNumber"] = cropped_code
                    st.success(f"Detected: {cat_num}")
                    st.rerun()
                else:
                    st.error("No barcode detected in crop box. Enter manually.")

            st.divider()

            st.markdown("#### 2. Vision AI Label Data")
            if st.button("🔄 Run / Re-Parse with Vision AI"):
                if not api_key:
                    st.error("Please enter a Gemini API Key in sidebar.")
                else:
                    with st.spinner("Analyzing sheet with Vision AI..."):
                        parsed = run_gemini_parser(image, api_key)
                        st.session_state.page_data[current_idx] = parsed
                        st.rerun()

            st.markdown("##### 🌿 Taxonomy")
            sci_name = st.text_input(
                "scientificName", value=pf.get("scientificName", "")
            )
            tax1, tax2 = st.columns(2)
            with tax1:
                genus = st.text_input("genus", value=pf.get("genus", ""))
            with tax2:
                sp_ep = st.text_input(
                    "specificEpithet", value=pf.get("specificEpithet", "")
                )

            st.markdown("##### 👤 Collector & Event Date")
            c1, c2 = st.columns(2)
            with c1:
                rec_by = st.text_input(
                    "recordedBy", value=pf.get("recordedBy", "")
                )
            with c2:
                rec_num = st.text_input(
                    "recordNumber", value=pf.get("recordNumber", "")
                )

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                ev_date = st.text_input(
                    "eventDate", value=pf.get("eventDate", "")
                )
            with d2:
                yr = st.text_input("year", value=pf.get("year", ""))
            with d3:
                mo = st.text_input("month", value=pf.get("month", ""))
            with d4:
                dy = st.text_input("day", value=pf.get("day", ""))

            st.markdown("##### 🗺️ Geography & Locality")
            g1, g2, g3, g4 = st.columns(4)
            with g1:
                cntry = st.text_input(
                    "country", value=pf.get("country", "United States")
                )
            with g2:
                state_prov = st.text_input(
                    "stateProvince", value=pf.get("stateProvince", "")
                )
            with g3:
                county = st.text_input("county", value=pf.get("county", ""))
            with g4:
                muni = st.text_input(
                    "municipality", value=pf.get("municipality", "")
                )

            locality = st.text_input("locality", value=pf.get("locality", ""))
            loc_rem = st.text_input(
                "locationRemarks", value=pf.get("locationRemarks", "")
            )

            st.markdown("##### 📍 Coordinates, Elevation & Phenology")
            loc_col1, loc_col2 = st.columns(2)
            with loc_col1:
                dec_lat = st.text_input(
                    "decimalLatitude", value=pf.get("decimalLatitude", "")
                )
                dec_lon = st.text_input(
                    "decimalLongitude", value=pf.get("decimalLongitude", "")
                )
                verb_coord = st.text_input(
                    "verbatimCoordinates",
                    value=pf.get("verbatimCoordinates", ""),
                )
            with loc_col2:
                min_elev = st.text_input(
                    "minimumElevationInMeters",
                    value=pf.get("minimumElevationInMeters", ""),
                )
                max_elev = st.text_input(
                    "maximumElevationInMeters",
                    value=pf.get("maximumElevationInMeters", ""),
                )
                verb_elev = st.text_input(
                    "verbatimElevation", value=pf.get("verbatimElevation", "")
                )

            symbiota_pheno_terms = [
                "",
                "In Flower",
                "In Fruit",
                "Flowering and Fruiting",
                "Flower Buds",
                "Vegetative",
                "Sterile",
                "Cones",
                "Spores",
            ]
            gemini_pheno_pred = pf.get("reproductiveCondition", "").strip()
            pheno_idx = next(
                (
                    i
                    for i, opt in enumerate(symbiota_pheno_terms)
                    if opt.lower() == gemini_pheno_pred.lower()
                ),
                0,
            )

            rep_cond = st.selectbox(
                "reproductiveCondition",
                options=symbiota_pheno_terms,
                index=pheno_idx,
            )

            st.markdown("##### 📝 Habitat, Substrate & Remarks")
            rcol1, rcol2 = st.columns(2)
            with rcol1:
                habitat = st.text_input("habitat", value=pf.get("habitat", ""))
                assoc_taxa = st.text_input(
                    "associatedTaxa", value=pf.get("associatedTaxa", "")
                )
            with rcol2:
                substrate = st.text_input(
                    "substrate", value=pf.get("substrate", "")
                )
                occ_rem = st.text_input(
                    "occurrenceRemarks", value=pf.get("occurrenceRemarks", "")
                )

            verb_label = st.text_area(
                "verbatimLabel", value=pf.get("verbatimLabel", ""), height=100
            )

            st.divider()

            if st.button("💾 Save Record & Next Specimen"):
                if not cat_num:
                    st.error("Catalog Number is required before saving.")
                else:
                    # Sanitize catalog number for filesystem safety
                    clean_cat = re.sub(r"[^\w\-]", "_", cat_num)
                    ext = os.path.splitext(img_path)[1]
                    new_filename = f"{clean_cat}{ext}"
                    dest_path = os.path.join(OUT_DIR, new_filename)

                    shutil.copy(img_path, dest_path)

                    rec_data = {
                        "institutionCode": inst_code,
                        "collectionCode": coll_code,
                        "catalogNumber": cat_num,
                        "scientificName": sci_name,
                        "genus": genus,
                        "specificEpithet": sp_ep,
                        "recordedBy": rec_by,
                        "recordNumber": rec_num,
                        "eventDate": ev_date,
                        "year": yr,
                        "month": mo,
                        "day": dy,
                        "occurrenceRemarks": occ_rem,
                        "habitat": habitat,
                        "substrate": substrate,
                        "associatedTaxa": assoc_taxa,
                        "reproductiveCondition": rep_cond,
                        "country": cntry,
                        "stateProvince": state_prov,
                        "county": county,
                        "municipality": muni,
                        "locality": locality,
                        "locationRemarks": loc_rem,
                        "decimalLatitude": dec_lat,
                        "decimalLongitude": dec_lon,
                        "verbatimCoordinates": verb_coord,
                        "minimumElevationInMeters": min_elev,
                        "maximumElevationInMeters": max_elev,
                        "verbatimElevation": verb_elev,
                        "verbatimLabel": verb_label,
                        "associatedMedia": new_filename,
                    }

                    st.session_state.records.append(rec_data)
                    st.session_state.page_data[current_idx] = rec_data.copy()
                    st.session_state.idx += 1
                    st.rerun()
    else:
        st.success("Batch processing complete!")

# Live Symbiota Export Spreadsheet
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
    st.session_state.records = edited_df.to_dict("records")

    st.markdown("### 🖼️ Saved Specimen Record Viewer")
    rec_labels = [
        f"Row {i+1}: [{r.get('catalogNumber', 'No-ID')}] {r.get('scientificName', 'Unidentified')}"
        for i, r in enumerate(st.session_state.records)
    ]
    selected_idx = st.selectbox(
        "Select a saved record to inspect:",
        options=range(len(rec_labels)),
        format_func=lambda x: rec_labels[x],
    )

    if selected_idx < len(st.session_state.records):
        rec = st.session_state.records[selected_idx]
        img_name = rec.get("associatedMedia", "")
        saved_img_path = os.path.join(OUT_DIR, img_name)

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
            st.json(rec)
else:
    st.info("No records saved in current session.")

# Sidebar Downloads
st.sidebar.header("4. Export Session Data")
if st.session_state.records:
    export_df = pd.DataFrame(st.session_state.records)
    # UTF-8 with BOM for native Windows Excel opening
    csv_bytes = export_df.to_csv(index=False, encoding="utf-8-sig").encode(
        "utf-8-sig"
    )

    st.sidebar.download_button(
        label="📥 Download Symbiota CSV",
        data=csv_bytes,
        file_name="symbiota_import.csv",
        mime="text/csv",
    )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for fname in os.listdir(OUT_DIR):
            fpath = os.path.join(OUT_DIR, fname)
            zf.write(fpath, fname)

    st.sidebar.download_button(
        label="📥 Download Renamed Images (.zip)",
        data=zip_buffer.getvalue(),
        file_name="renamed_specimens.zip",
        mime="application/zip",
    )
