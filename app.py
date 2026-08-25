import io
import json
import os
import shutil
import tempfile
import time
import zipfile
import cv2
import folium
from geopy.geocoders import ArcGIS
from google import genai
from google.genai import types
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
from pyzbar.pyzbar import decode
import requests
import streamlit as st
from streamlit_cropper import st_cropper
from streamlit_folium import st_folium
import zxingcpp

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION, STICKY CSS & SECURITY GATE
# -----------------------------------------------------------------------------
st.set_page_config(
    layout="wide", page_title="Herbarium Image-First Digitization (WSCO)"
)

st.markdown(
    """
    <style>
        [data-testid="stHorizontalBlock"] > div:first-child {
            position: sticky;
            top: 2rem;
            align-self: flex-start;
            max-height: 92vh;
            overflow-y: auto;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

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

st.sidebar.header("1. Institutional Defaults")
inst_code = st.sidebar.text_input("institutionCode", value="Weber State")
coll_code = st.sidebar.text_input("collectionCode", value="WSCO")

st.sidebar.header("2. Vision AI Configuration")
user_api_key = st.sidebar.text_input(
    "Gemini API Key (Optional)",
    type="password",
    help="Default key is loaded from secrets if available.",
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
# 2. DISK CACHE & SESSION PURGE ENGINE
# -----------------------------------------------------------------------------
AUTOSAVE_FILE = os.path.join(tempfile.gettempdir(), "wsco_herbarium_autosave.json")


def save_autosave():
    """Writes session state to disk cache."""
    try:
        data = {
            "records": st.session_state.get("records", []),
            "idx": st.session_state.get("idx", 0),
            "page_data": {
                str(k): v
                for k, v in st.session_state.get("page_data", {}).items()
            },
            "image_paths": st.session_state.get("image_paths", []),
        }
        with open(AUTOSAVE_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def purge_all_session_data():
    """Completely wipes session memory and temporary disk image directories."""
    if os.path.exists(AUTOSAVE_FILE):
        try:
            os.remove(AUTOSAVE_FILE)
        except Exception:
            pass

    if "work_dir" in st.session_state and os.path.exists(st.session_state.work_dir):
        shutil.rmtree(st.session_state.work_dir, ignore_errors=True)
    if "out_dir" in st.session_state and os.path.exists(st.session_state.out_dir):
        shutil.rmtree(st.session_state.out_dir, ignore_errors=True)

    st.session_state.work_dir = tempfile.mkdtemp()
    st.session_state.out_dir = tempfile.mkdtemp()
    st.session_state.records = []
    st.session_state.idx = 0
    st.session_state.page_data = {}
    st.session_state.image_paths = []


if "work_dir" not in st.session_state:
    st.session_state.work_dir = tempfile.mkdtemp()
if "out_dir" not in st.session_state:
    st.session_state.out_dir = tempfile.mkdtemp()

if "records" not in st.session_state:
    if os.path.exists(AUTOSAVE_FILE):
        try:
            with open(AUTOSAVE_FILE, "r") as f:
                saved = json.load(f)
            st.session_state.records = saved.get("records", [])
            st.session_state.idx = saved.get("idx", 0)
            st.session_state.page_data = {
                int(k): v for k, v in saved.get("page_data", {}).items()
            }
            st.session_state.image_paths = saved.get("image_paths", [])
        except Exception:
            purge_all_session_data()
    else:
        st.session_state.records = []
        st.session_state.idx = 0
        st.session_state.page_data = {}
        st.session_state.image_paths = []


# -----------------------------------------------------------------------------
# 3. PYDANTIC STRUCTURED OUTPUT SCHEMA
# -----------------------------------------------------------------------------
class HerbariumSchema(BaseModel):
    catalogNumber: str = Field(default="", description="Extracted barcode or catalog ID number")
    barcodeBox: list[int] = Field(default_factory=list, description="[ymin, xmin, ymax, xmax] scaled 0-1000")
    labelBox: list[int] = Field(default_factory=list, description="[ymin, xmin, ymax, xmax] scaled 0-1000")
    scientificName: str = Field(default="", description="Full taxon name including author if present")
    genus: str = Field(default="", description="Genus name")
    specificEpithet: str = Field(default="", description="Species epithet")
    scientificNameAuthorship: str = Field(default="", description="Author name string")
    identifiedBy: str = Field(default="", description="Determiner name")
    recordedBy: str = Field(default="", description="Collector name(s)")
    associatedCollectors: str = Field(default="", description="Co-collectors list")
    recordNumber: str = Field(default="", description="Collector field number")
    eventDate: str = Field(default="", description="Collection date in YYYY-MM-DD format if possible")
    verbatimEventDate: str = Field(default="", description="Raw date string as written")
    year: str = Field(default="", description="4-digit year")
    month: str = Field(default="", description="Numeric month (1-12)")
    day: str = Field(default="", description="Numeric day (1-31)")
    occurrenceRemarks: str = Field(default="", description="Plant description or specimen observations")
    habitat: str = Field(default="", description="Habitat or community notes")
    substrate: str = Field(default="", description="Soil or growing medium notes")
    associatedTaxa: str = Field(default="", description="Associated species list")
    reproductiveCondition: str = Field(default="", description="Exact match from phenology terms or empty string")
    country: str = Field(default="United States", description="Country name, EXACTLY 'United States' if US")
    stateProvince: str = Field(default="", description="State or Province")
    county: str = Field(default="", description="County or Parish")
    municipality: str = Field(default="", description="City, town, or municipality")
    locality: str = Field(default="", description="Detailed locality description")
    geocodingSearchTerm: str = Field(default="", description="Simplified landmark name for geocoding")
    locationRemarks: str = Field(default="", description="Additional location notes")
    decimalLatitude: str = Field(default="", description="Numeric latitude in decimal degrees or empty string")
    decimalLongitude: str = Field(default="", description="Numeric longitude in decimal degrees or empty string")
    verbatimCoordinates: str = Field(default="", description="Raw coordinate string from label")
    elevationNumber: str = Field(default="", description="Single elevation value in meters or empty string if range")
    minimumElevationInMeters: str = Field(default="", description="Lower elevation in meters if range, else empty string")
    maximumElevationInMeters: str = Field(default="", description="Upper elevation in meters if range, else empty string")
    verbatimElevation: str = Field(default="", description="Raw elevation string as recorded on label")
    verbatimLabel: str = Field(default="", description="Full exact verbatim label text")


DEFAULT_DWC_RECORD = HerbariumSchema().model_dump()
DEFAULT_DWC_RECORD["coordinateUncertaintyInMeters"] = "1000"
DEFAULT_DWC_RECORD["geodeticDatum"] = "WGS84"


# -----------------------------------------------------------------------------
# 4. HELPER FUNCTIONS & GEOPROCESSING
# -----------------------------------------------------------------------------
def crop_box_1000(img: Image.Image, box) -> Image.Image:
    if not box:
        return None
    try:
        w, h = img.size
        if isinstance(box, dict) and "left" in box:
            left = int(box["left"])
            top = int(box["top"])
            right = int(box["left"] + box["width"])
            bottom = int(box["top"] + box["height"])
            return img.crop((left, top, right, bottom))
        elif isinstance(box, list) and len(box) == 4:
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


def georeference_geolocate(locality: str, state: str, county: str):
    if not locality:
        return None
    url = "https://www.geo-locate.org/webservices/geolocatesvcs/glws.asmx/Georef2"
    clean_county = (
        county.replace(" County", "").replace(" Co.", "").strip()
        if county
        else ""
    )
    params = {
        "locality": locality,
        "state": state or "",
        "county": clean_county,
        "country": "United States",
        "fmt": "json",
    }
    try:
        resp = requests.get(url, params=params, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            points = data.get("engine", {}).get("result", {}).get("resultSet", [])
            if points:
                top = points[0]
                lat = str(round(float(top["WGS84Latitude"]), 6))
                lon = str(round(float(top["WGS84Longitude"]), 6))
                unc = str(int(top.get("UncertaintyRadiusmeters", 2500)))
                return lat, lon, unc
    except Exception:
        pass
    return None


def georeference_arcgis(search_term: str, county: str, state: str):
    if not search_term and not county:
        return None
    try:
        geolocator = ArcGIS(user_agent="weber_state_herbarium_digitizer")
        query_parts = [p for p in [search_term, county, state, "United States"] if p]
        query = ", ".join(query_parts)
        loc = geolocator.geocode(query, timeout=5)
        if loc:
            return str(round(loc.latitude, 6)), str(round(loc.longitude, 6)), "3000"
    except Exception:
        pass
    return None


def run_gemini_parser(img: Image.Image, key: str) -> dict:
    """Optimized Vision AI Parser: Fast payload downscaling & direct fast model execution."""
    if not key:
        res = DEFAULT_DWC_RECORD.copy()
        res["verbatimLabel"] = "Error: Missing Gemini API Key."
        return res

    try:
        # 1. Downscale image payload for fast network transfer (keeps 100% OCR clarity)
        img_payload = img.convert("RGB")
        img_payload.thumbnail((1600, 1600), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        img_payload.save(buf, format="JPEG", quality=80)
        image_part = types.Part.from_bytes(
            data=buf.getvalue(),
            mime_type="image/jpeg"
        )

        client = genai.Client(api_key=key)

        prompt = (
            "You are an expert botanical taxonomist and herbarium digitizer. "
            "Examine this specimen sheet and extract all visible text into the requested JSON schema. "
            "Extract primary label data, scientific name, collector, dates, location, and barcode numbers."
        )

        # 2. Direct fast endpoints (skips slow client.models.list network call)
        primary_models = ["gemini-2.5-flash", "gemini-2.0-flash"]
        
        parsed_data = None
        last_error = None
        response = None

        for model_name in primary_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, image_part],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=HerbariumSchema,
                        temperature=0.1,
                    ),
                )

                if hasattr(response, "parsed") and response.parsed:
                    if isinstance(response.parsed, BaseModel):
                        parsed_data = response.parsed.model_dump()
                        break
                    elif isinstance(response.parsed, dict):
                        parsed_data = response.parsed
                        break

                if hasattr(response, "text") and response.text:
                    clean_text = response.text.replace("```json", "").replace("```", "").strip()
                    if clean_text:
                        parsed_data = json.loads(clean_text)
                        break

            except Exception as err:
                last_error = err
                err_str = str(err)
                if "404" in err_str or "NOT_FOUND" in err_str:
                    continue  # Try next model immediately if deprecated
                time.sleep(1)

        if parsed_data:
            merged = DEFAULT_DWC_RECORD.copy()
            merged.update(parsed_data)

            if response and hasattr(response, "usage_metadata") and response.usage_metadata:
                merged["_inputTokens"] = str(getattr(response.usage_metadata, "prompt_token_count", 0) or 0)
                merged["_outputTokens"] = str(getattr(response.usage_metadata, "candidates_token_count", 0) or 0)
                merged["_totalTokens"] = str(getattr(response.usage_metadata, "total_token_count", 0) or 0)

            return merged

        raise ValueError(f"All model attempts failed. Last error: {last_error}")

    except Exception as e:
        res = DEFAULT_DWC_RECORD.copy()
        res["verbatimLabel"] = f"API Error: {str(e)}"
        st.error(f"⚠️ Gemini API Execution Failed: {str(e)}")
        return res


def render_map_georeference(specimen_data, record_key="specimen"):
    lat_val = str(specimen_data.get("decimalLatitude", "")).strip()
    lon_val = str(specimen_data.get("decimalLongitude", "")).strip()

    if not lat_val or not lon_val:
        locality = specimen_data.get("locality", "").strip()
        state = specimen_data.get("stateProvince", "").strip()
        county = specimen_data.get("county", "").strip()
        search_term = specimen_data.get("geocodingSearchTerm", "").strip()

        gl_res = georeference_geolocate(locality, state, county)
        if gl_res:
            specimen_data["decimalLatitude"] = gl_res[0]
            specimen_data["decimalLongitude"] = gl_res[1]
            specimen_data["coordinateUncertaintyInMeters"] = gl_res[2]
        else:
            ag_res = georeference_arcgis(search_term or locality, county, state)
            if ag_res:
                specimen_data["decimalLatitude"] = ag_res[0]
                specimen_data["decimalLongitude"] = ag_res[1]
                specimen_data["coordinateUncertaintyInMeters"] = ag_res[2]

    has_coords = False
    try:
        lat = float(specimen_data.get("decimalLatitude"))
        lon = float(specimen_data.get("decimalLongitude"))
        has_coords = True
    except (ValueError, TypeError):
        lat, lon = 39.8283, -98.5795

    try:
        uncertainty = float(
            specimen_data.get("coordinateUncertaintyInMeters", 1000)
        )
    except (ValueError, TypeError):
        uncertainty = 1000.0

    m = folium.Map(location=[lat, lon], zoom_start=11 if has_coords else 4)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap",
        name="Topographic",
    ).add_to(m)

    if has_coords:
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
        lat_click = round(map_response["last_clicked"]["lat"], 6)
        lon_click = round(map_response["last_clicked"]["lng"], 6)
        specimen_data["decimalLatitude"] = str(lat_click)
        specimen_data["decimalLongitude"] = str(lon_click)

    c1, c2, c3 = st.columns(3)
    with c1:
        in_lat = st.text_input(
            "Latitude",
            value=specimen_data.get("decimalLatitude", ""),
            key=f"lat_in_{record_key}",
        )
        specimen_data["decimalLatitude"] = in_lat.strip()
    with c2:
        in_lon = st.text_input(
            "Longitude",
            value=specimen_data.get("decimalLongitude", ""),
            key=f"lon_in_{record_key}",
        )
        specimen_data["decimalLongitude"] = in_lon.strip()
    with c3:
        in_unc = st.text_input(
            "Uncertainty (m)",
            value=str(int(uncertainty)),
            key=f"unc_in_{record_key}",
        )
        specimen_data["coordinateUncertaintyInMeters"] = in_unc.strip()


# -----------------------------------------------------------------------------
# 5. BATCH UPLOAD MANAGEMENT & PURGE CONTROL
# -----------------------------------------------------------------------------
st.sidebar.header("3. Upload Batch")

if st.sidebar.button("🧹 Clear All Previous Images & Start Fresh"):
    purge_all_session_data()
    st.rerun()

uploaded_files = st.sidebar.file_uploader(
    "Upload ZIP archive or images (JPG, PNG, TIF)",
    type=["zip", "jpg", "jpeg", "png", "tif", "tiff"],
    accept_multiple_files=True,
    key="specimen_uploader",
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
    save_autosave()
    st.rerun()

# -----------------------------------------------------------------------------
# 6. MAIN DIGITIZATION WORKSPACE
# -----------------------------------------------------------------------------
if st.session_state.image_paths:
    total_imgs = len(st.session_state.image_paths)

    if st.session_state.idx < total_imgs:
        img_path = st.session_state.image_paths[st.session_state.idx]
        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image)
        current_idx = st.session_state.idx

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
            save_autosave()

        pf = st.session_state.page_data[current_idx]

        nav1, nav2, nav3, nav4 = st.columns([2, 1, 1, 1])
        with nav1:
            st.markdown(
                f"### Specimen {current_idx + 1} of {total_imgs}: `{os.path.basename(img_path)}`"
            )
        with nav2:
            if st.button("⬅️ Previous") and current_idx > 0:
                st.session_state.idx -= 1
                save_autosave()
                st.rerun()
        with nav3:
            if st.button("⏭️ Skip") and current_idx < total_imgs - 1:
                st.session_state.idx += 1
                save_autosave()
                st.rerun()
        with nav4:
            if st.button("↩️ Undo Last") and st.session_state.records:
                last_rec = st.session_state.records.pop()
                last_f = os.path.join(
                    st.session_state.out_dir,
                    last_rec.get("associatedMedia", ""),
                )
                if os.path.exists(last_f):
                    os.remove(last_f)
                st.session_state.idx = max(0, st.session_state.idx - 1)
                save_autosave()
                st.rerun()

        st.divider()

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.caption("Full Specimen Sheet (Pinned). Draw a crop box if manual barcode cropping is needed.")
            barcode_box = st_cropper(
                image,
                realtime_update=True,
                box_color="#0000FF",
                key=f"cropper_{current_idx}",
                return_type="box",
            )

            b_crop = crop_box_1000(image, pf.get("barcodeBox"))
            l_crop = crop_box_1000(image, pf.get("labelBox"))

            if b_crop or l_crop:
                st.markdown("#### 🔎 AI Zoomed Previews")
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

        with col_right:
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
                    save_autosave()
                    st.success(f"Detected Barcode: {cat_num}")
                else:
                    st.error("No barcode detected in crop box. Enter manually.")

            st.divider()

            st.markdown("#### 2. Vision AI Label Data")

            in_tok = pf.get("_inputTokens", "0")
            out_tok = pf.get("_outputTokens", "0")
            tot_tok = pf.get("_totalTokens", "0")

            st.info(
                f"📊 **Token Metrics:** Prompt/Image: `{in_tok}` tokens | Output: `{out_tok}` tokens | **Total:** `{tot_tok}` tokens"
            )

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
                        save_autosave()
                        st.rerun()

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

                st.caption("📍 Interactive Georeferencing Map (GEOLocate & ArcGIS)")
                render_map_georeference(pf, record_key=f"rec_{current_idx}")

                verb_coord = st.text_input(
                    "verbatimCoordinates",
                    value=pf.get("verbatimCoordinates", ""),
                )

                lcol1, lcol2, lcol3 = st.columns(3)
                with lcol1:
                    elev_num = st.text_input(
                        "elevationNumber",
                        value=pf.get("elevationNumber", ""),
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
                    st.error("Catalog Number (Barcode) is required before saving.")
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
                        "elevationNumber": elev_num,
                        "minimumElevationInMeters": min_elev,
                        "maximumElevationInMeters": max_elev,
                        "verbatimElevation": verb_elev,
                        "verbatimLabel": verb_label,
                        "associatedMedia": new_filename,
                    }

                    st.session_state.records.append(rec_data)
                    st.session_state.page_data[current_idx] = rec_data.copy()
                    st.session_state.idx += 1
                    save_autosave()
                    st.rerun()
    else:
        st.success("🎉 Batch processing complete!")

# -----------------------------------------------------------------------------
# 7. SPREADSHEET EDITOR & EXPORTS
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
    save_autosave()

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
