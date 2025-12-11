import os
import sys
import time
from typing import Optional, Tuple
import io
import base64

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import streamlit as st
from PIL import Image
import tempfile

# Directory where text outputs are saved
SAVE_DIR = "scanned_texts"
os.makedirs(SAVE_DIR, exist_ok=True)


def setup_tesseract_path() -> None:
    """Auto-detect and configure Tesseract path on Windows."""
    if sys.platform == "win32":
        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv("USERNAME", "")),
        ]
        for path in common_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return


def enhance_image_quality(image: np.ndarray) -> np.ndarray:
    """Enhance image quality for better OCR results."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    min_dimension = min(h, w)
    if min_dimension < 300:
        scale_factor = 300.0 / min_dimension
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    return enhanced


def preprocess_image(image: np.ndarray, method: str = "combined") -> np.ndarray:
    """Apply advanced preprocessing to boost OCR accuracy."""
    enhanced = enhance_image_quality(image)
    
    if method == "adaptive":
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif method == "otsu":
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "morphology":
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    else:  # combined
        adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.sum(adaptive == 255) > np.sum(otsu == 255):
            thresh = adaptive
        else:
            thresh = otsu
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return thresh


def deskew_image(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """Detect and correct skew in the image."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    if lines is None or len(lines) == 0:
        return image, 0.0
    
    angles = []
    for line in lines[:20]:
        rho, theta = line[0]  # HoughLines returns array of shape (N, 1, 2)
        angle = (theta * 180 / np.pi) - 90
        if -45 < angle < 45:
            angles.append(angle)
    
    if not angles:
        return image, 0.0
    
    median_angle = np.median(angles)
    if abs(median_angle) < 0.5:
        return image, 0.0
    
    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return corrected, median_angle


def run_ocr_with_multiple_modes(image: np.ndarray, preprocessing_method: str = "combined") -> Tuple[str, dict, str]:
    """Run OCR with multiple PSM modes and return the best result."""
    processed = preprocess_image(image, method=preprocessing_method)
    
    psm_modes = [
        ("6", "Uniform block of text"),
        ("3", "Fully automatic page segmentation"),
        ("11", "Sparse text"),
        ("12", "Sparse text with OSD"),
        ("7", "Single text line"),
        ("8", "Single word"),
    ]
    
    best_text = ""
    best_data = {}
    best_psm = "6"
    best_confidence = 0.0
    
    for psm, description in psm_modes:
        try:
            config = f"--psm {psm}"
            text = pytesseract.image_to_string(processed, config=config)
            data = pytesseract.image_to_data(processed, output_type=Output.DICT, config=config)
            
            confidences = [float(c) for c in data.get("conf", []) if c not in ("", "-1")]
            avg_conf = np.mean(confidences) if confidences else 0.0
            score = avg_conf * len(text.strip())
            
            if score > best_confidence:
                best_confidence = score
                best_text = text
                best_data = data
                best_psm = psm
        except Exception:
            continue
    
    if not best_text.strip():
        config = "--psm 6"
        best_text = pytesseract.image_to_string(processed, config=config)
        best_data = pytesseract.image_to_data(processed, output_type=Output.DICT, config=config)
        best_psm = "6"
    
    return best_text, best_data, best_psm


def post_process_text(text: str) -> str:
    """Clean and improve extracted text."""
    if not text:
        return ""
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = ' '.join(line.split())
        if len(line) >= 2:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def draw_ocr_overlay(image: np.ndarray, ocr_data: dict, conf_threshold: int = 50) -> np.ndarray:
    """Draw bounding boxes and recognized words on top of the image."""
    overlay = image.copy()
    n = len(ocr_data.get("text", []))

    for i in range(n):
        text = ocr_data["text"][i].strip()
        conf = int(float(ocr_data["conf"][i])) if ocr_data["conf"][i] not in ("", "-1") else -1
        if not text or conf < conf_threshold:
            continue

        x, y, w, h = (
            ocr_data["left"][i],
            ocr_data["top"][i],
            ocr_data["width"][i],
            ocr_data["height"][i],
        )
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            overlay,
            text,
            (x, y - 5 if y - 5 > 0 else y + h + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return overlay


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image."""
    if len(image.shape) == 2:
        return Image.fromarray(image)
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            return Image.fromarray(image)
    return Image.fromarray(image)


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array."""
    return np.array(image)


def prepare_image_for_display(image: np.ndarray) -> np.ndarray:
    """Convert image to RGB format for Streamlit display."""
    if len(image.shape) == 2:
        # Grayscale image - convert to RGB
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            # BGR image - convert to RGB
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            # RGBA image - convert to RGB
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return image


def crop_roi(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Crop region of interest from image."""
    h, w = image.shape[:2]
    x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
    x2, y2 = max(x1, min(x2, w)), max(y1, min(y2, h))
    return image[y1:y2, x1:x2]


def main():
    st.set_page_config(
        page_title="Printed Text Scanner - Web App",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📄 Printed Text Scanner - Web Application")
    st.markdown("---")
    
    # Setup Tesseract
    try:
        setup_tesseract_path()
    except:
        pass
    
    # Initialize session state
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'roi_coords' not in st.session_state:
        st.session_state.roi_coords = None
    if 'ocr_result' not in st.session_state:
        st.session_state.ocr_result = None
    if 'overlay_image' not in st.session_state:
        st.session_state.overlay_image = None
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        preprocessing_method = st.selectbox(
            "Preprocessing Method",
            ["Combined (Best)", "Adaptive", "Otsu", "Morphology"],
            index=0,
            help="Choose the image preprocessing method for OCR"
        )
        
        enable_deskew = st.checkbox(
            "Auto Deskew",
            value=True,
            help="Automatically detect and correct image rotation"
        )
        
        st.markdown("---")
        st.header("📤 Image Input")
        
        input_method = st.radio(
            "Choose input method:",
            ["Upload Image", "Webcam Capture"],
            index=0
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Image Preview")
        
        # Image input
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tif'],
                help="Upload an image containing text to scan"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.session_state.current_image = pil_to_numpy(image)
        else:  # Webcam
            st.info("💡 Webcam capture: Use your device's camera to take a photo")
            camera_input = st.camera_input("Take a picture")
            
            if camera_input is not None:
                image = Image.open(camera_input)
                st.session_state.current_image = pil_to_numpy(image)
        
        # Display current image
        if st.session_state.current_image is not None:
            display_image = st.session_state.current_image.copy()
            
            # Show ROI selection if enabled
            st.markdown("**Select Region of Interest (Optional)**")
            use_roi = st.checkbox("Enable ROI Selection", value=False)
            
            if use_roi:
                st.info("💡 Click and drag on the image below to select a region")
                img_pil = numpy_to_pil(display_image)
                
                # Use streamlit-image-coordinates for ROI selection
                # For now, we'll use manual coordinate input
                st.write("Enter ROI coordinates (or use full image):")
                col_x1, col_y1, col_x2, col_y2 = st.columns(4)
                
                h, w = display_image.shape[:2]
                
                with col_x1:
                    x1 = st.number_input("X1", min_value=0, max_value=w, value=0, key="x1")
                with col_y1:
                    y1 = st.number_input("Y1", min_value=0, max_value=h, value=0, key="y1")
                with col_x2:
                    x2 = st.number_input("X2", min_value=0, max_value=w, value=w, key="x2")
                with col_y2:
                    y2 = st.number_input("Y2", min_value=0, max_value=h, value=h, key="y2")
                
                if x2 > x1 and y2 > y1:
                    st.session_state.roi_coords = (x1, y1, x2, y2)
                    # Draw ROI rectangle on image
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 255), 3)
                else:
                    st.warning("⚠️ Invalid ROI coordinates!")
                    st.session_state.roi_coords = None
            else:
                st.session_state.roi_coords = None
            
            # Display image (convert BGR to RGB for Streamlit)
            display_rgb = prepare_image_for_display(display_image)
            st.image(display_rgb, width='stretch')
            
            # OCR button
            if st.button("🔍 Run OCR", type="primary", use_container_width=True):
                with st.spinner("Processing image with OCR..."):
                    try:
                        image_to_process = st.session_state.current_image.copy()
                        
                        # Apply deskewing if enabled
                        if enable_deskew:
                            image_to_process, skew_angle = deskew_image(image_to_process)
                            if abs(skew_angle) > 0.5:
                                st.info(f"📐 Image deskewed by {skew_angle:.2f} degrees")
                        
                        # Extract ROI if selected
                        if st.session_state.roi_coords:
                            x1, y1, x2, y2 = st.session_state.roi_coords
                            roi_image = crop_roi(image_to_process, x1, y1, x2, y2)
                        else:
                            roi_image = image_to_process
                        
                        # Map preprocessing method
                        method_map = {
                            "Combined (Best)": "combined",
                            "Adaptive": "adaptive",
                            "Otsu": "otsu",
                            "Morphology": "morphology"
                        }
                        method = method_map.get(preprocessing_method, "combined")
                        
                        # Run OCR
                        text, ocr_data, used_psm = run_ocr_with_multiple_modes(roi_image, method)
                        cleaned_text = post_process_text(text)
                        
                        # Create overlay
                        overlay = draw_ocr_overlay(roi_image, ocr_data, conf_threshold=50)
                        
                        # Merge overlay back to original if ROI was used
                        if st.session_state.roi_coords:
                            x1, y1, x2, y2 = st.session_state.roi_coords
                            result_image = image_to_process.copy()
                            h_roi, w_roi = roi_image.shape[:2]
                            result_image[y1:y1+h_roi, x1:x1+w_roi] = overlay
                            st.session_state.overlay_image = result_image
                        else:
                            st.session_state.overlay_image = overlay
                        
                        st.session_state.ocr_result = cleaned_text
                        
                        st.success(f"✅ OCR completed! (PSM Mode: {used_psm}, Method: {preprocessing_method})")
                        
                    except pytesseract.pytesseract.TesseractNotFoundError:
                        st.error("""
                        ❌ **Tesseract OCR not found!**
                        
                        Please install Tesseract OCR:
                        1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
                        2. Install to default location
                        3. Restart this application
                        """)
                    except Exception as e:
                        st.error(f"❌ Error during OCR: {str(e)}")
            
            # Show overlay if available
            if st.session_state.overlay_image is not None:
                st.subheader("🎨 OCR Overlay Preview")
                overlay_rgb = prepare_image_for_display(st.session_state.overlay_image)
                st.image(overlay_rgb, width='stretch')
    
    with col2:
        st.subheader("📝 Extracted Text")
        
        if st.session_state.ocr_result:
            st.text_area(
                "OCR Result",
                st.session_state.ocr_result,
                height=400,
                key="ocr_text_display"
            )
            
            # Download button
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"scanned_text_{timestamp}.txt"
            st.download_button(
                label="💾 Download Text",
                data=st.session_state.ocr_result,
                file_name=filename,
                mime="text/plain",
                use_container_width=True
            )
            
            # Save to file button
            if st.button("💾 Save to File", use_container_width=True):
                filepath = os.path.join(SAVE_DIR, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(st.session_state.ocr_result)
                st.success(f"✅ Text saved to: {filepath}")
        else:
            st.info("👆 Upload an image and click 'Run OCR' to extract text")
            st.text_area(
                "OCR Result",
                "",
                height=400,
                disabled=True,
                placeholder="Extracted text will appear here..."
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Printed Text Scanner Web App | Enhanced OCR with Multiple Preprocessing Methods</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

