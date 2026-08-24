import io
import json
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
    layout="wide", page_title="Herbarium Digitization Suite (WSCO)"
)

# Password Protection Gate
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

# API Key Configuration (Falls back safely to Streamlit Cloud Secrets)
st.sidebar.header("Configuration & API Key")
user_api_key = st.sidebar.text_input(
    "Custom Gemini API Key (Optional)",
    type="password",
    help="Leave blank to use the pre-configured institutional default key.",
)
API_KEY = (
    user_api_key.strip()
    if user_api_key.strip()
    else st.secrets.get("GEMINI_API_KEY", "")
)

# -----------------------------------------------------------------------------
# 2. DARWIN CORE SCHEMA & HELPER FUNCTIONS
# -----------------------------------------------------------------------------
DEFAULT_DWC_RECORD = {
    "institutionCode": "WSCO",
    "catalogNumber": "",
    "basisOfRecord": "PreservedSpecimen",
    "scientificName": "",
    "family": "",
    "scientificNameAuthorship": "",
    "identifiedBy": "",
    "dateIdentified": "",
    "identificationQualifier": "",
    "recordedBy": "",
    "associatedCollectors": "",
    "recordNumber": "",
    "eventDate": "",
    "verbatimEventDate": "",
    "country": "United States",
    "stateProvince": "Utah",
    "county": "",
    "locality": "",
    "verbatimCoordinates": "",
    "decimalLatitude": "",
    "decimalLongitude": "",
    "coordinateUncertaintyInMeters": "1000",
    "geodeticDatum": "WGS84",
    "verbatimElevation": "",
    "minimumElevationInMeters": "",
    "habitat": "",
    "associatedTaxa": "",
    "reproductiveCondition": "",
    "occurrenceRemarks": "",
}


def read_barcode_from_image(pil_img):
    """Scans image for standard barcodes using zxingcpp with pyzbar fallback."""
    img_np = np.array(pil_img)
    try:
        results = zxingcpp.read_barcodes(img_np)
        if results:
            return results[0].text
    except Exception:
        pass

    try:
        barcodes = decode(img_np)
        if barcodes:
            return barcodes[0].data.decode("utf-8")
    except Exception:
        pass

    return ""


def extract_dwc_with_gemini(crop_pil, api_key):
    """Sends cropped label image to Gemini VLM for Darwin Core parsing."""
    if not api_key:
        st.error(
            "Gemini API key is missing. Add it to Streamlit Secrets or sidebar."
        )
        return {}

    client = genai.Client(api_key=api_key)
    prompt = """
    Analyze this herbarium specimen label image. Extract all printed/written text and map it into 
    a strictly structured JSON object following Darwin Core standard terms:
    
    Keys to extract:
    - scientificName (Genus species without author)
    - family (Plant family name)
    - scientificNameAuthorship (Taxon author name string)
    - identifiedBy (Determiner name)
    - dateIdentified (Date of identification)
    - recordedBy (Primary collector name)
    - associatedCollectors (Co-collectors list)
    - recordNumber (Collector field number)
    - eventDate (Collection date in YYYY-MM-DD format if possible)
    - verbatimEventDate (Exact collection date as written)
    - country (Country name)
    - stateProvince (State/Province name)
    - county (County name)
    - locality (Detailed text description of location)
    - verbatimCoordinates (TRS, UTM, or raw lat/lon text)
    - verbatimElevation (Elevation as written, e.g. '5400 ft')
    - habitat (Ecological habitat notes, slope, aspect, soil)
    - associatedTaxa (Associated species names listed)
    - reproductiveCondition (Phenology: flowering, fruiting, sterile)
    - occurrenceRemarks (Plant description: height, flower color, frequency)

    Return ONLY a valid JSON object.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[crop_pil, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Gemini Extraction Error: {str(e)}")
        return {}


def render_map_georeference(specimen_data, record_key="specimen"):
    """Auto-geocodes locality text and renders an interactive verification map with uncertainty radius."""
    # 1. Auto-geocode if coordinates are missing but locality text exists
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

    # 2. Extract Lat, Lon, Uncertainty defaults (Default location: Ogden, UT)
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

    # 3. Build Interactive Folium Map
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

    # 4. Map Click Event: Update lat/lon when user clicks terrain
    if map_response and map_response.get("last_clicked"):
        lat = round(map_response["last_clicked"]["lat"], 6)
        lon = round(map_response["last_clicked"]["lng"], 6)
        specimen_data["decimalLatitude"] = str(lat)
        specimen_data["decimalLongitude"] = str(lon)

    # 5. Coordinate Input Controls
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
# 3. MAIN APPLICATION & WORKFLOW
# -----------------------------------------------------------------------------
st.title("🌿 WSCO Herbarium Digitization & Georeferencing")

if "records" not in st.session_state:
    st.session_state.records = []
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0

# File Upload Section
uploaded_files = st.sidebar.file_uploader(
    "Upload Specimen Images",
    type=["jpg", "jpeg", "png", "tif"],
    accept_multiple_files=True,
)

if uploaded_files:
    if len(st.session_state.records) != len(uploaded_files):
        st.session_state.records = [
            {**DEFAULT_DWC_RECORD, "filename": f.name} for f in uploaded_files
        ]

    # Batch Navigation Controls
    st.sidebar.subheader("Batch Progress")
    st.session_state.current_idx = st.sidebar.number_input(
        f"Specimen Record (1 of {len(uploaded_files)})",
        min_value=1,
        max_value=len(uploaded_files),
        value=st.session_state.current_idx + 1,
        step=1,
    ) - 1

    curr_file = uploaded_files[st.session_state.current_idx]
    curr_record = st.session_state.records[st.session_state.current_idx]

    image = Image.open(curr_file)
    image = ImageOps.exif_transpose(image)

    # Main Workspace Layout
    col_img, col_form = st.columns([1.1, 1.2])

    with col_img:
        st.subheader("1. Label & Barcode Cropper")

        # Auto-detect barcode on load if catalogNumber is empty
        if not curr_record["catalogNumber"]:
            detected_barcode = read_barcode_from_image(image)
            if detected_barcode:
                curr_record["catalogNumber"] = detected_barcode
                st.success(f"Auto-detected Barcode: {detected_barcode}")

        st.caption("Crop the main label box below to extract text with Gemini:")
        cropped_img = st_cropper(
            image,
            realtime_update=True,
            box_color="#3186cc",
            key=f"crop_{st.session_state.current_idx}",
        )

        if st.button("✨ Run Gemini AI Extraction", type="primary"):
            with st.spinner("Extracting Darwin Core text..."):
                extracted = extract_dwc_with_gemini(cropped_img, API_KEY)
                for key, val in extracted.items():
                    if key in curr_record and val:
                        curr_record[key] = str(val)
                st.rerun()

    with col_form:
        st.subheader("2. Darwin Core Verification Form")

        tabs = st.tabs(
            ["Catalog & Taxon", "Event & Collector", "Locality & Map", "Notes"]
        )

        with tabs[0]:
            c1, c2 = st.columns(2)
            with c1:
                curr_record["catalogNumber"] = st.text_input(
                    "Catalog Number (Barcode)",
                    value=curr_record["catalogNumber"],
                )
                curr_record["scientificName"] = st.text_input(
                    "Scientific Name", value=curr_record["scientificName"]
                )
                curr_record["family"] = st.text_input(
                    "Family", value=curr_record["family"]
                )
            with c2:
                curr_record["institutionCode"] = st.text_input(
                    "Institution Code", value=curr_record["institutionCode"]
                )
                curr_record["scientificNameAuthorship"] = st.text_input(
                    "Author", value=curr_record["scientificNameAuthorship"]
                )
                curr_record["identifiedBy"] = st.text_input(
                    "Identified By", value=curr_record["identifiedBy"]
                )

        with tabs[1]:
            c1, c2 = st.columns(2)
            with c1:
                curr_record["recordedBy"] = st.text_input(
                    "Primary Collector", value=curr_record["recordedBy"]
                )
                curr_record["associatedCollectors"] = st.text_input(
                    "Associated Collectors",
                    value=curr_record["associatedCollectors"],
                )
                curr_record["recordNumber"] = st.text_input(
                    "Collector Number", value=curr_record["recordNumber"]
                )
            with c2:
                curr_record["eventDate"] = st.text_input(
                    "Event Date (YYYY-MM-DD)", value=curr_record["eventDate"]
                )
                curr_record["verbatimEventDate"] = st.text_input(
                    "Verbatim Date", value=curr_record["verbatimEventDate"]
                )

        with tabs[2]:
            curr_record["locality"] = st.text_area(
                "Locality Description",
                value=curr_record["locality"],
                height=70,
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                curr_record["country"] = st.text_input(
                    "Country", value=curr_record["country"]
                )
            with c2:
                curr_record["stateProvince"] = st.text_input(
                    "State/Province", value=curr_record["stateProvince"]
                )
            with c3:
                curr_record["county"] = st.text_input(
                    "County", value=curr_record["county"]
                )

            st.caption("📍 Georeferencing Map (Click map to position pin)")
            render_map_georeference(
                curr_record, record_key=f"rec_{st.session_state.current_idx}"
            )

            c1, c2 = st.columns(2)
            with c1:
                curr_record["verbatimCoordinates"] = st.text_input(
                    "Verbatim Coordinates / TRS",
                    value=curr_record["verbatimCoordinates"],
                )
            with c2:
                curr_record["verbatimElevation"] = st.text_input(
                    "Verbatim Elevation",
                    value=curr_record["verbatimElevation"],
                )

        with tabs[3]:
            curr_record["habitat"] = st.text_area(
                "Habitat", value=curr_record["habitat"], height=60
            )
            curr_record["associatedTaxa"] = st.text_input(
                "Associated Taxa", value=curr_record["associatedTaxa"]
            )
            curr_record["reproductiveCondition"] = st.text_input(
                "Reproductive Condition",
                value=curr_record["reproductiveCondition"],
            )
            curr_record["occurrenceRemarks"] = st.text_area(
                "Occurrence Remarks",
                value=curr_record["occurrenceRemarks"],
                height=60,
            )

    # Export Section
    st.divider()
    st.subheader("3. Export Symbiota CSV")
    df_export = pd.DataFrame(st.session_state.records)
    st.dataframe(df_export, height=150, use_container_width=True)

    csv_data = df_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Symbiota Darwin Core CSV",
        data=csv_data,
        file_name="WSCO_Herbarium_Export.csv",
        mime="text/csv",
        type="primary",
    )
else:
    st.info("Please upload specimen images in the sidebar to begin batch digitizing.")
