import os
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PyQt5 import QtCore, QtGui, QtWidgets


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
        # If not found, show helpful error
        error_msg = (
            "Tesseract OCR not found!\n\n"
            "Please install Tesseract OCR:\n"
            "1. Download from: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "2. Install to default location (C:\\Program Files\\Tesseract-OCR)\n"
            "3. Or set the path manually by editing gui_app.py\n\n"
            "If already installed, add it to your PATH environment variable."
        )
        print(error_msg)
        raise FileNotFoundError("Tesseract OCR executable not found. Please install Tesseract.")


def enhance_image_quality(image: np.ndarray) -> np.ndarray:
    """Enhance image quality for better OCR results."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Scale up image if too small (OCR works better with higher resolution)
    h, w = gray.shape
    min_dimension = min(h, w)
    scale_factor = 1.0
    if min_dimension < 300:
        scale_factor = 300.0 / min_dimension
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Denoise using bilateral filter (preserves edges while removing noise)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    return enhanced


def preprocess_image(image: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """
    Apply advanced preprocessing to boost OCR accuracy.
    
    Args:
        image: Input image (BGR or grayscale)
        method: Preprocessing method - 'adaptive', 'otsu', 'morphology', or 'combined'
    """
    # First enhance image quality
    enhanced = enhance_image_quality(image)
    
    if method == "adaptive":
        # Adaptive thresholding - works well for varying lighting
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif method == "otsu":
        # Otsu's thresholding - automatic threshold selection
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "morphology":
        # Morphological operations to clean up the image
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Remove noise
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    else:  # combined - best of all methods
        # Try adaptive first
        adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        # Also get Otsu
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Use the one with better contrast (more white pixels usually means better)
        if np.sum(adaptive == 255) > np.sum(otsu == 255):
            thresh = adaptive
        else:
            thresh = otsu
        # Clean up with morphology
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return thresh


def deskew_image(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """Detect and correct skew in the image."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Hough line transform to detect skew angle
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    if lines is None or len(lines) == 0:
        return image, 0.0
    
    angles = []
    for rho, theta in lines[:20]:  # Check first 20 lines
        angle = (theta * 180 / np.pi) - 90
        if -45 < angle < 45:
            angles.append(angle)
    
    if not angles:
        return image, 0.0
    
    # Get median angle (more robust than mean)
    median_angle = np.median(angles)
    
    # Only correct if angle is significant (> 0.5 degrees)
    if abs(median_angle) < 0.5:
        return image, 0.0
    
    # Rotate image to correct skew
    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return corrected, median_angle


def run_ocr_with_multiple_modes(image: np.ndarray, preprocessing_method: str = "combined") -> Tuple[str, dict, str]:
    """
    Run OCR with multiple PSM modes and return the best result.
    
    Returns:
        Tuple of (best_text, best_ocr_data, best_psm_mode)
    """
    # Preprocess image
    processed = preprocess_image(image, method=preprocessing_method)
    
    # Try multiple PSM modes
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
            # Use PSM mode without character whitelist for better accuracy
            # You can add whitelist if needed: -c tessedit_char_whitelist=...
            config = f"--psm {psm}"
            text = pytesseract.image_to_string(processed, config=config)
            data = pytesseract.image_to_data(processed, output_type=Output.DICT, config=config)
            
            # Calculate average confidence
            confidences = [float(c) for c in data.get("conf", []) if c not in ("", "-1")]
            avg_conf = np.mean(confidences) if confidences else 0.0
            
            # Prefer results with higher confidence and more text
            score = avg_conf * len(text.strip())
            
            if score > best_confidence:
                best_confidence = score
                best_text = text
                best_data = data
                best_psm = psm
        except Exception:
            continue
    
    # If no good result, fall back to default
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
        # Remove excessive whitespace
        line = ' '.join(line.split())
        # Remove lines that are too short (likely noise)
        if len(line) >= 2:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def draw_ocr_overlay(image: np.ndarray, ocr_data: dict, conf_threshold: int = 60) -> np.ndarray:
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


def to_qimage(frame: np.ndarray) -> QtGui.QImage:
    """Convert BGR numpy image to QImage."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)


def timestamped_filename() -> str:
    return os.path.join(SAVE_DIR, f"scanned_text_{time.strftime('%Y%m%d-%H%M%S')}.txt")


class ROIImageLabel(QtWidgets.QLabel):
    """QLabel that supports click-and-drag rectangle selection."""

    roiSelected = QtCore.pyqtSignal(QtCore.QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background: #111; border: 1px solid #444;")
        self._pixmap: Optional[QtGui.QPixmap] = None
        self._start: Optional[QtCore.QPoint] = None
        self._end: Optional[QtCore.QPoint] = None
        self.selection_enabled = False

    def setPixmap(self, pm: QtGui.QPixmap) -> None:  # type: ignore[override]
        self._pixmap = pm
        super().setPixmap(pm)

    def enable_selection(self, enabled: bool) -> None:
        self.selection_enabled = enabled
        if not enabled:
            self._start = None
            self._end = None
            self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self.selection_enabled or event.button() != QtCore.Qt.LeftButton:
            return
        self._start = event.pos()
        self._end = event.pos()
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self.selection_enabled or self._start is None:
            return
        self._end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self.selection_enabled or self._start is None or self._end is None:
            return
        rect = QtCore.QRect(self._start, self._end).normalized()
        self.roiSelected.emit(rect)
        self.enable_selection(False)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        if self._start and self._end and self.selection_enabled:
            painter = QtGui.QPainter(self)
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 180, 255), 2, QtCore.Qt.DashLine))
            painter.drawRect(QtCore.QRect(self._start, self._end))
            painter.end()


class OCRApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Printed Text Scanner GUI - Enhanced OCR")
        self.resize(1200, 700)
        
        # Add status bar
        self.statusBar().showMessage("Ready")

        self.cap: Optional[cv2.VideoCapture] = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.current_frame: Optional[np.ndarray] = None
        self.frozen_image: Optional[np.ndarray] = None
        self.selected_roi: Optional[QtCore.QRect] = None
        self.preprocessing_method = "combined"  # Default preprocessing method
        self.enable_deskew = True  # Enable deskewing by default

        self._build_ui()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        self.video_label = ROIImageLabel()
        self.video_label.setMinimumHeight(400)
        self.video_label.roiSelected.connect(self.on_roi_selected)

        self.text_display = QtWidgets.QPlainTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setPlaceholderText("Extracted text will appear here...")

        # Preprocessing options
        preprocess_label = QtWidgets.QLabel("Preprocessing:")
        self.preprocess_combo = QtWidgets.QComboBox()
        self.preprocess_combo.addItems(["Combined (Best)", "Adaptive", "Otsu", "Morphology"])
        self.preprocess_combo.setCurrentText("Combined (Best)")
        self.preprocess_combo.currentTextChanged.connect(self.on_preprocess_changed)
        
        self.deskew_checkbox = QtWidgets.QCheckBox("Auto Deskew")
        self.deskew_checkbox.setChecked(True)
        self.deskew_checkbox.stateChanged.connect(self.on_deskew_changed)

        # Buttons
        self.btn_start_cam = QtWidgets.QPushButton("Start Camera")
        self.btn_stop_cam = QtWidgets.QPushButton("Stop Camera")
        self.btn_load = QtWidgets.QPushButton("Load Image")
        self.btn_capture = QtWidgets.QPushButton("Capture Frame")
        self.btn_select_roi = QtWidgets.QPushButton("Select ROI")
        self.btn_run_ocr = QtWidgets.QPushButton("Run OCR")
        self.btn_save_text = QtWidgets.QPushButton("Save Text")
        self.btn_clear = QtWidgets.QPushButton("Clear")

        self.btn_stop_cam.setEnabled(False)
        self.btn_capture.setEnabled(False)
        self.btn_select_roi.setEnabled(False)
        self.btn_run_ocr.setEnabled(False)
        self.btn_save_text.setEnabled(False)

        # Options layout
        options_layout = QtWidgets.QHBoxLayout()
        options_layout.addWidget(preprocess_label)
        options_layout.addWidget(self.preprocess_combo)
        options_layout.addWidget(self.deskew_checkbox)
        options_layout.addStretch()

        button_layout = QtWidgets.QGridLayout()
        button_layout.addWidget(self.btn_start_cam, 0, 0)
        button_layout.addWidget(self.btn_stop_cam, 0, 1)
        button_layout.addWidget(self.btn_load, 0, 2)
        button_layout.addWidget(self.btn_capture, 0, 3)
        button_layout.addWidget(self.btn_select_roi, 1, 0)
        button_layout.addWidget(self.btn_run_ocr, 1, 1)
        button_layout.addWidget(self.btn_save_text, 1, 2)
        button_layout.addWidget(self.btn_clear, 1, 3)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.video_label, stretch=2)
        layout.addLayout(options_layout)
        layout.addLayout(button_layout)
        layout.addWidget(self.text_display, stretch=1)

        central.setLayout(layout)

        # Signals
        self.btn_start_cam.clicked.connect(self.start_camera)
        self.btn_stop_cam.clicked.connect(self.stop_camera)
        self.btn_load.clicked.connect(self.load_image)
        self.btn_capture.clicked.connect(self.capture_frame)
        self.btn_select_roi.clicked.connect(self.enable_roi_selection)
        self.btn_run_ocr.clicked.connect(self.run_ocr)
        self.btn_save_text.clicked.connect(self.save_text)
        self.btn_clear.clicked.connect(self.clear_all)

    # Camera handling -----------------------------------------------------
    def start_camera(self) -> None:
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Camera Error", "Unable to access camera.")
            self.cap = None
            return
        self.timer.start(30)
        self.btn_start_cam.setEnabled(False)
        self.btn_stop_cam.setEnabled(True)
        self.btn_capture.setEnabled(True)
        self.btn_select_roi.setEnabled(False)
        self.btn_run_ocr.setEnabled(False)

    def stop_camera(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.btn_start_cam.setEnabled(True)
        self.btn_stop_cam.setEnabled(False)
        self.btn_capture.setEnabled(False)
        self.btn_select_roi.setEnabled(self.frozen_image is not None)
        self.btn_run_ocr.setEnabled(self.frozen_image is not None)

    def update_frame(self) -> None:
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_frame = frame
        self.display_image(frame)

    # Image utilities -----------------------------------------------------
    def display_image(self, image: np.ndarray) -> None:
        self.current_frame = image
        qimg = to_qimage(image)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    def capture_frame(self) -> None:
        if self.current_frame is None:
            return
        self.frozen_image = self.current_frame.copy()
        self.stop_camera()
        self.btn_select_roi.setEnabled(True)
        self.btn_run_ocr.setEnabled(True)
        self.display_image(self.frozen_image)

    def load_image(self) -> None:
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif)"
        )
        if not fname:
            return
        image = cv2.imread(fname)
        if image is None:
            QtWidgets.QMessageBox.warning(self, "Load Error", "Could not load the selected image.")
            return
        self.frozen_image = image
        self.stop_camera()
        self.display_image(image)
        self.btn_select_roi.setEnabled(True)
        self.btn_run_ocr.setEnabled(True)

    def enable_roi_selection(self) -> None:
        if self.frozen_image is None:
            return
        self.selected_roi = None
        self.video_label.enable_selection(True)

    def on_roi_selected(self, rect: QtCore.QRect) -> None:
        self.selected_roi = rect

    def on_preprocess_changed(self, text: str) -> None:
        """Handle preprocessing method change."""
        method_map = {
            "Combined (Best)": "combined",
            "Adaptive": "adaptive",
            "Otsu": "otsu",
            "Morphology": "morphology"
        }
        self.preprocessing_method = method_map.get(text, "combined")

    def on_deskew_changed(self, state: int) -> None:
        """Handle deskew checkbox change."""
        self.enable_deskew = state == QtCore.Qt.Checked

    # OCR -----------------------------------------------------------------
    def run_ocr(self) -> None:
        if self.frozen_image is None:
            QtWidgets.QMessageBox.information(self, "No Image", "Capture or load an image first.")
            return

        # Show processing message
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.btn_run_ocr.setEnabled(False)
        self.btn_run_ocr.setText("Processing...")
        QtWidgets.QApplication.processEvents()

        try:
            image = self.frozen_image.copy()
            roi = self._extract_roi(image) if self.selected_roi else image
            if roi.size == 0:
                QtWidgets.QMessageBox.warning(self, "ROI Error", "Selected ROI is empty.")
                return

            # Apply deskewing if enabled
            if self.enable_deskew:
                roi, skew_angle = deskew_image(roi)
                if abs(skew_angle) > 0.5 and not self.selected_roi:
                    # Update the full image if deskewed and no ROI selected
                    image = roi.copy()

            # Run OCR with multiple modes and enhanced preprocessing
            text, ocr_data, used_psm = run_ocr_with_multiple_modes(roi, self.preprocessing_method)
            
            # Post-process text
            cleaned_text = post_process_text(text)
            
            # Draw overlay
            overlay = draw_ocr_overlay(roi, ocr_data, conf_threshold=50)  # Lower threshold for more detections
            
            # Display results
            self.display_image(self._merge_overlay(image, overlay))
            self.text_display.setPlainText(cleaned_text)
            self.btn_save_text.setEnabled(bool(cleaned_text.strip()))
            
            # Show info about which mode was used
            if cleaned_text.strip():
                status_text = f"OCR completed (PSM mode: {used_psm}, Method: {self.preprocessing_method})"
                self.statusBar().showMessage(status_text, 3000)
            
        except pytesseract.pytesseract.TesseractNotFoundError:
            error_msg = (
                "Tesseract OCR not found!\n\n"
                "Please install Tesseract OCR:\n"
                "1. Download from: https://github.com/UB-Mannheim/tesseract/wiki\n"
                "2. Install to default location (C:\\Program Files\\Tesseract-OCR)\n"
                "3. Restart this application\n\n"
                "If already installed, ensure it's in your PATH or edit gui_app.py to set the path manually."
            )
            QtWidgets.QMessageBox.critical(self, "Tesseract Not Found", error_msg)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "OCR Error", f"An error occurred during OCR:\n{str(e)}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.btn_run_ocr.setEnabled(True)
            self.btn_run_ocr.setText("Run OCR")

    def _extract_roi(self, image: np.ndarray) -> np.ndarray:
        """Convert the QRect ROI into image crop coordinates."""
        h, w, _ = image.shape
        label_w = self.video_label.width()
        label_h = self.video_label.height()
        if self.selected_roi is None or label_w == 0 or label_h == 0:
            return image

        # Calculate scaling ratios between displayed pixmap and original image
        pixmap = self.video_label._pixmap
        if pixmap is None or pixmap.width() == 0 or pixmap.height() == 0:
            return image

        scale = min(label_w / pixmap.width(), label_h / pixmap.height())
        displayed_w = int(pixmap.width() * scale)
        displayed_h = int(pixmap.height() * scale)

        offset_x = (label_w - displayed_w) // 2
        offset_y = (label_h - displayed_h) // 2

        x1 = max(0, int((self.selected_roi.left() - offset_x) / scale))
        y1 = max(0, int((self.selected_roi.top() - offset_y) / scale))
        x2 = min(w, int((self.selected_roi.right() - offset_x) / scale))
        y2 = min(h, int((self.selected_roi.bottom() - offset_y) / scale))

        return image[y1:y2, x1:x2]

    def _merge_overlay(self, original: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        """Place overlay back on the original image if ROI was used."""
        if self.selected_roi is None:
            return overlay

        h, w, _ = original.shape
        roi = self._extract_roi(original)
        if roi.shape[:2] != overlay.shape[:2]:
            return overlay

        x, y, _, _ = self._roi_image_coords(original.shape)
        if x is None:
            return overlay
        result = original.copy()
        h_o, w_o, _ = overlay.shape
        result[y : y + h_o, x : x + w_o] = overlay
        return result

    def _roi_image_coords(self, shape: Tuple[int, int, int]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """Return ROI coordinates in image space."""
        h, w, _ = shape
        label_w = self.video_label.width()
        label_h = self.video_label.height()
        pixmap = self.video_label._pixmap
        if self.selected_roi is None or pixmap is None:
            return None, None, None, None
        scale = min(label_w / pixmap.width(), label_h / pixmap.height())
        displayed_w = int(pixmap.width() * scale)
        displayed_h = int(pixmap.height() * scale)
        offset_x = (label_w - displayed_w) // 2
        offset_y = (label_h - displayed_h) // 2
        x1 = max(0, int((self.selected_roi.left() - offset_x) / scale))
        y1 = max(0, int((self.selected_roi.top() - offset_y) / scale))
        x2 = min(w, int((self.selected_roi.right() - offset_x) / scale))
        y2 = min(h, int((self.selected_roi.bottom() - offset_y) / scale))
        return x1, y1, x2, y2

    # Persistence ---------------------------------------------------------
    def save_text(self) -> None:
        text = self.text_display.toPlainText().strip()
        if not text:
            QtWidgets.QMessageBox.information(self, "No Text", "Nothing to save yet.")
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Extracted Text", timestamped_filename(), "Text Files (*.txt)"
        )
        if not filename:
            return
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        QtWidgets.QMessageBox.information(self, "Saved", f"Text saved to:\n{filename}")

    def clear_all(self) -> None:
        self.text_display.clear()
        self.selected_roi = None
        self.frozen_image = None
        self.btn_save_text.setEnabled(False)
        self.btn_run_ocr.setEnabled(False)
        self.btn_select_roi.setEnabled(False)
        if self.current_frame is not None:
            blank = np.zeros_like(self.current_frame)
            self.display_image(blank)

    # Cleanup -------------------------------------------------------------
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        self.stop_camera()
        super().closeEvent(event)


def main() -> None:
    # Try to configure Tesseract path (Windows)
    try:
        setup_tesseract_path()
    except FileNotFoundError:
        # Continue anyway - error will be shown when user tries OCR
        pass
    
    app = QtWidgets.QApplication(sys.argv)
    window = OCRApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

