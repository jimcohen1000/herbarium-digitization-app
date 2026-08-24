import io
import json
import os
import shutil
import tempfile
import zipfile
import cv2
import folium
from geopy.geocoders import Nominatim
from google import genai
from google.genai import types
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from pyzbar.pyzbar import decode
import streamlit as st
from streamlit_cropper import st_cropper
from streamlit_folium import st_folium
import zxingcpp

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION & SECURITY GATE
# -----------------------------------------------------------------------------
st.set_page_config(
    layout="wide", page_title="Herbarium Image-First Digitization (WSCO)"
)

# App Password Security Gate
expected_password = st.secrets.get("APP_PASSWORD")
if expected_password:
    user_password = st.sidebar.text_input(
        "Herbarium Team Password", type="password"
    )
    if user_password != expected_password:
        st.warning(
            "Please enter the team password in the sidebar to access the app."
        )
        st.stop()

# Institutional & API Key Settings
st.sidebar.header("1. Institutional Defaults")
inst_code = st.sidebar.text_input("institutionCode", value="Weber State")
coll_code = st.sidebar.text_input("collectionCode", value="WSCO")

st.sidebar.header("2. Vision AI Configuration")
user_api_key = st.sidebar.text_input(
    "Gemini API Key (Optional)",
    type="password",
    help="Default key is loaded from secrets if available. Clear and type to override.",
)
API_KEY = (
    user_api_key.strip()
    if user_api_key.strip()
    else st.secrets.get("GEMINI_API_KEY", "")
)

auto_parse = st.sidebar.checkbox(
    "⚡ Auto-parse image on load",
    value=False,
    help="Automatically run Vision AI whenever a new specimen image loads.",
)

# -----------------------------------------------------------------------------
# 2. SESSION STATE & DEFAULT SCHEMA
# -----------------------------------------------------------------------------
if "records" not in st.session_state:
    st.session_state.records = []
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "page_data" not in st.session_state:
    st.session_state.page_data = {}
if "image_paths" not in st.session_state:
    st.session_state.image_paths = []
if "work_dir" not in st.session_state:
    st.session_state.work_dir = tempfile.mkdtemp()
if "out_dir" not in st.session_state:
    st.session_state.out_dir = tempfile.mkdtemp()

DEFAULT_DWC_RECORD = {
    "catalogNumber": "",
    "barcodeBox": [],
    "labelBox": [],
    "scientificName": "",
    "genus": "",
    "specificEpithet": "",
    "scientificNameAuthorship": "",
    "identifiedBy": "",
    "dateIdentified": "",
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
    "country": "United States",
    "stateProvince": "Utah",
    "county": "",
    "municipality": "",
    "locality": "",
    "locationRemarks": "",
    "decimalLatitude": "",
    "decimalLongitude": "",
    "coordinateUncertaintyInMeters": "1000",
    "verbatimCoordinates": "",
    "geodeticDatum": "WGS84",
    "minimumElevationInMeters": "",
    "maximumElevationInMeters": "",
    "verbatimElevation": "",
    "verbatimLabel": "",
}


# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS: CROPPING, BARCODES, GEMINI & MAPS
# -----------------------------------------------------------------------------
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


def decode_barcode_fullres(img: Image.Image, crop_box: dict = None) -> str:
    """Slices original full-res pixels using relative crop coordinates and applies adaptive thresholding."""
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


def run_gemini_parser(img: Image.Image, key: str) -> dict:
    """Uses Google GenAI SDK to parse full sheet images with detailed Darwin Core prompt rules."""
    if not key:
        res = DEFAULT_DWC_RECORD.copy()
        res["verbatimLabel"] = "Error: Missing Gemini API Key."
        return res

    try:
        client = genai.Client(api_key=key)
        candidate_models = [
            "gemini-2.5-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
        ]

        prompt = """
        Examine this herbarium specimen sheet. Locate the primary specimen label, plant specimen, and barcode sticker.
        Extract data into standard Symbiota / Darwin Core fields.

        STRICT STANDARDIZATION RULES:
        1. COUNTRY: If the specimen is from the US, set "country" to EXACTLY "United States" (do NOT use "US", "USA", or "United States of America").
        2. TAXONOMY: Extract "scientificName" (e.g. Pinus ponderosa Douglas ex C.Lawson). Break out "genus" (e.g. Pinus) and "specificEpithet" (e.g. ponderosa).
        3. DATES: Extract "eventDate" (e.g. 1984-06-15). Also break out integer values for "year", "month", and "day" separately. Extract "verbatimEventDate" as written.
        4. COORDINATES: Extract "verbatimCoordinates" as printed. Convert any DMS or UTM into decimal degrees as "decimalLatitude" and "decimalLongitude". Ensure West and South values are negative numbers.
        5. ELEVATION: Extract "verbatimElevation" as printed. Convert numeric values to meters.
           - If a single value is given (e.g., 1500 m or 5000 ft), set BOTH "minimumElevationInMeters" AND "maximumElevationInMeters" to that converted value in meters.
           - If a range is given (e.g., 1500-1800 m), set min and max accordingly.
        6. PHENOLOGY: Inspect the plant material for flowers, fruits, or cones. Assign "reproductiveCondition" to EXACTLY one of:
           ["In Flower", "In Fruit", "Flowering and Fruiting", "Flower Buds", "Vegetative", "Sterile", "Cones", "Spores"].
           - If unclear, unidentifiable, or ambiguous, leave as empty string ("").

        Return ONLY a JSON object matching this schema:
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
            "substrate": "Soil, rock type, or growing medium notes",
            "associatedTaxa": "Associated species list",
            "reproductiveCondition": "Exact match from phenology terms or empty string",
            "country": "Country name (must be 'United States' for US specimens)",
            "stateProvince": "State or Province",
            "county": "County or Parish",
            "municipality": "City, town, or municipality",
            "locality": "Detailed locality description",
            "locationRemarks": "Additional location or access notes",
            "decimalLatitude": "Numeric latitude in decimal degrees",
            "decimalLongitude": "Numeric longitude in decimal degrees",
            "verbatimCoordinates": "Raw coordinate string from label",
            "minimumElevationInMeters": "Min elevation in meters",
            "maximumElevationInMeters": "Max elevation in meters",
            "verbatimElevation": "Raw elevation string as recorded on label",
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
                    parsed = json.loads(response.text)
                    merged = DEFAULT_DWC_RECORD.copy()
                    merged.update(parsed)
                    return merged
            except Exception as err:
                last_error = err
                continue

        if last_error:
            raise last_error

    except Exception as e:
        res = DEFAULT_DWC_RECORD.copy()
        res["verbatimLabel"] = f"API Error: {str(e)}"
        return res


def render_map_georeference(specimen_data, record_key="specimen"):
    """Auto-geocodes locality text and renders an interactive verification map with uncertainty radius."""
    if not specimen_data.get("decimalLatitude") and specimen_data.get(
        "locality"
    ):
        try:
            geolocator = Nominatim(user_agent="weber_state_herbarium_digitizer")
            q_parts = [
                specimen_data.get("locality", ""),
                specimen_data.get("county", ""),
                specimen_data.get("stateProvince", ""),
            ]
            query = ", ".join([p for p in q_parts if p])
            location = geolocator.geocode(query, timeout=4)
            if location:
                specimen_data["decimalLatitude"] = str(
                    round(location.latitude, 6)
                )
                specimen_data["decimalLongitude"] = str(
                    round(location.longitude, 6)
                )
        except Exception:
            pass

    try:
        lat = float(specimen_data.get("decimalLatitude", 41.2230))
        lon = float(specimen_data.get("decimalLongitude", -111.9738))
    except (ValueError, TypeError):
        lat, lon = 41.2230, -111.9738

    try:
        uncertainty = float(
            specimen_data.get("coordinateUncertaintyInMeters", 1000)
        )
    except (ValueError, TypeError):
        uncertainty = 1000.0

    m = folium.Map(location=[lat, lon], zoom_start=11)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap",
        name="Topographic",
    ).add_to(m)

    folium.Marker(
        [lat, lon],
        popup=f"Coordinates: {lat}, {lon}",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    folium.Circle(
        location=[lat, lon],
        radius=uncertainty,
        color="#3186cc",
        fill=True,
        fill_color="#3186cc",
        fill_opacity=0.2,
        popup=f"Uncertainty: {int(uncertainty)} m",
    ).add_to(m)

    map_response = st_folium(
        m, height=280, use_container_width=True, key=f"map_{record_key}"
    )

    if map_response and map_response.get("last_clicked"):
        lat = round(map_response["last_clicked"]["lat"], 6)
        lon = round(map_response["last_clicked"]["lng"], 6)
        specimen_data["decimalLatitude"] = str(lat)
        specimen_data["decimalLongitude"] = str(lon)

    c1, c2, c3 = st.columns(3)
    with c1:
        specimen_data["decimalLatitude"] = st.text_input(
            "Latitude", value=str(lat), key=f"lat_in_{record_key}"
        )
    with c2:
        specimen_data["decimalLongitude"] = st.text_input(
            "Longitude", value=str(lon), key=f"lon_in_{record_key}"
        )
    with c3:
        specimen_data["coordinateUncertaintyInMeters"] = st.text_input(
            "Uncertainty (m)",
            value=str(int(uncertainty)),
            key=f"unc_in_{record_key}",
        )


# -----------------------------------------------------------------------------
# 4. BATCH UPLOAD MANAGEMENT
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 5. MAIN DIGITIZATION WORKSPACE
# -----------------------------------------------------------------------------
if st.session_state.image_paths:
    total_imgs = len(st.session_state.image_paths)

    if st.session_state.idx < total_imgs:
        img_path = st.session_state.image_paths[st.session_state.idx]
        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image)
        current_idx = st.session_state.idx

        # Initialize or fetch cached page data
        if current_idx not in st.session_state.page_data:
            if auto_parse and API_KEY:
                with st.spinner(
                    f"🤖 Auto-parsing specimen {current_idx + 1} with Vision AI..."
                ):
                    st.session_state.page_data[current_idx] = run_gemini_parser(
                        image, API_KEY
                    )
            else:
                st.session_state.page_data[current_idx] = (
                    DEFAULT_DWC_RECORD.copy()
                )

        pf = st.session_state.page_data[current_idx]

        # Navigation Bar
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
                last_f = os.path.join(
                    st.session_state.out_dir, last_rec.get("associatedMedia", "")
                )
                if os.path.exists(last_f):
                    os.remove(last_f)
                st.session_state.idx = max(0, st.session_state.idx - 1)
                st.rerun()

        st.divider()

        col_left, col_right = st.columns([1, 1])

        # Left Column: Image Viewer & Dynamic Cropper
        with col_left:
            st.caption(
                "Full Specimen Sheet. Draw a blue crop box over barcode if manual detection is required."
            )
            barcode_box = st_cropper(
                image,
                realtime_update=True,
                box_color="#0000FF",
                key=f"cropper_{current_idx}",
                return_type="box",
            )

        # Right Column: AI Crops & Multi-Tab Darwin Core Editing
        with col_right:
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

            # Barcode Identification Controls
            st.markdown("#### 1. Barcode Identification")
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
                    pf["catalogNumber"] = cat_num
                    st.success(f"Detected Barcode: {cat_num}")
                else:
                    st.error("No barcode detected in crop box. Enter manually.")

            st.divider()

            # Vision AI Execution Button
            st.markdown("#### 2. Vision AI Label Data")
            if st.button("🔄 Run / Re-Parse with Vision AI", type="primary"):
                if not API_KEY:
                    st.error(
                        "Please enter a valid Gemini API Key in the sidebar or secrets."
                    )
                else:
                    with st.spinner("Analyzing sheet with Vision AI..."):
                        st.session_state.page_data[current_idx] = (
                            run_gemini_parser(image, API_KEY)
                        )
                        st.rerun()

            # Darwin Core Verification Tabs
            tabs = st.tabs(
                [
                    "Taxonomy",
                    "Collector & Dates",
                    "Locality & Map",
                    "Remarks & Phenology",
                ]
            )

            with tabs[0]:
                sci_name = st.text_input(
                    "scientificName", value=pf.get("scientificName", "")
                )
                tcol1, tcol2 = st.columns(2)
                with tcol1:
                    genus = st.text_input("genus", value=pf.get("genus", ""))
                with tcol2:
                    sp_ep = st.text_input(
                        "specificEpithet", value=pf.get("specificEpithet", "")
                    )

                tcol3, tcol4 = st.columns(2)
                with tcol3:
                    author = st.text_input(
                        "scientificNameAuthorship",
                        value=pf.get("scientificNameAuthorship", ""),
                    )
                with tcol4:
                    id_by = st.text_input(
                        "identifiedBy", value=pf.get("identifiedBy", "")
                    )

            with tabs[1]:
                c1, c2 = st.columns(2)
                with c1:
                    rec_by = st.text_input(
                        "recordedBy", value=pf.get("recordedBy", "")
                    )
                    assoc_coll = st.text_input(
                        "associatedCollectors",
                        value=pf.get("associatedCollectors", ""),
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

                verb_date = st.text_input(
                    "verbatimEventDate", value=pf.get("verbatimEventDate", "")
                )

            with tabs[2]:
                locality = st.text_area(
                    "locality", value=pf.get("locality", ""), height=70
                )
                loc_rem = st.text_input(
                    "locationRemarks", value=pf.get("locationRemarks", "")
                )

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

                st.caption(
                    "📍 Interactive Georeferencing Map (Click terrain to set pin)"
                )
                render_map_georeference(pf, record_key=f"rec_{current_idx}")

                lcol1, lcol2, lcol3 = st.columns(3)
                with lcol1:
                    verb_coord = st.text_input(
                        "verbatimCoordinates",
                        value=pf.get("verbatimCoordinates", ""),
                    )
                with lcol2:
                    min_elev = st.text_input(
                        "minimumElevationInMeters",
                        value=pf.get("minimumElevationInMeters", ""),
                    )
                with lcol3:
                    max_elev = st.text_input(
                        "maximumElevationInMeters",
                        value=pf.get("maximumElevationInMeters", ""),
                    )

                verb_elev = st.text_input(
                    "verbatimElevation", value=pf.get("verbatimElevation", "")
                )

            with tabs[3]:
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

                rcol1, rcol2 = st.columns(2)
                with rcol1:
                    habitat = st.text_input(
                        "habitat", value=pf.get("habitat", "")
                    )
                    assoc_taxa = st.text_input(
                        "associatedTaxa", value=pf.get("associatedTaxa", "")
                    )
                with rcol2:
                    substrate = st.text_input(
                        "substrate", value=pf.get("substrate", "")
                    )
                    occ_rem = st.text_input(
                        "occurrenceRemarks",
                        value=pf.get("occurrenceRemarks", ""),
                    )

                verb_label = st.text_area(
                    "verbatimLabel", value=pf.get("verbatimLabel", ""), height=90
                )

            st.divider()

            if st.button("💾 Save Record & Next Specimen", type="primary"):
                if not cat_num:
                    st.error(
                        "Catalog Number (Barcode) is required before saving."
                    )
                else:
                    ext = os.path.splitext(img_path)[1]
                    new_filename = f"{cat_num}{ext}"
                    dest_path = os.path.join(
                        st.session_state.out_dir, new_filename
                    )

                    shutil.copy(img_path, dest_path)

                    rec_data = {
                        "institutionCode": inst_code,
                        "collectionCode": coll_code,
                        "catalogNumber": cat_num,
                        "scientificName": sci_name,
                        "genus": genus,
                        "specificEpithet": sp_ep,
                        "scientificNameAuthorship": author,
                        "identifiedBy": id_by,
                        "recordedBy": rec_by,
                        "associatedCollectors": assoc_coll,
                        "recordNumber": rec_num,
                        "eventDate": ev_date,
                        "verbatimEventDate": verb_date,
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
                        "decimalLatitude": pf.get("decimalLatitude", ""),
                        "decimalLongitude": pf.get("decimalLongitude", ""),
                        "coordinateUncertaintyInMeters": pf.get(
                            "coordinateUncertaintyInMeters", "1000"
                        ),
                        "geodeticDatum": "WGS84",
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
        st.success("🎉 Batch processing complete!")

# -----------------------------------------------------------------------------
# 6. LIVE SPREADSHEET EDITOR & EXPORT SUITE
# -----------------------------------------------------------------------------
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

    st.markdown("### 🖼️ Saved Specimen Viewer")
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
                    caption=f"Renamed Specimen File: {img_name}",
                    use_container_width=True,
                )
            else:
                st.warning("Saved image file not found.")
        with rcol2:
            st.markdown("**Saved Darwin Core Fields**")
            st.json(rec)
else:
    st.info(
        "No saved records in current session. Upload images and process specimens to build your export."
    )

# Sidebar Exports
st.sidebar.header("4. Export Session Data")
if st.session_state.records:
    export_df = pd.DataFrame(st.session_state.records)
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")

    st.sidebar.download_button(
        label="📥 Download Symbiota CSV",
        data=csv_bytes,
        file_name="WSCO_Symbiota_Import.csv",
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
