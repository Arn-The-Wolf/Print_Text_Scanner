# GitHub Repository Description Options

## Short Description (for GitHub "About" section - 160 characters max)
```
📄 Desktop & Web OCR Scanner with PyTesseract. Extract text from images with advanced preprocessing, ROI selection, and live camera support.
```

## Medium Description (for GitHub Topics/Tags)
```
A comprehensive OCR (Optical Character Recognition) application with both desktop (PyQt5) and web (Streamlit) interfaces. Features advanced image preprocessing, multi-mode OCR, auto deskewing, ROI selection, and real-time text extraction from images and camera feeds.
```

## Full Description (for README or detailed documentation)

### What This Project Does

**Printed Text Scanner** is a dual-interface OCR application that extracts text from images using advanced computer vision and machine learning techniques. The project provides both a desktop GUI application (PyQt5) and a web-based interface (Streamlit), making it accessible for various use cases.

### Key Capabilities

- **📷 Multiple Input Methods**: Upload images, capture from webcam, or use live camera feed
- **🔍 Advanced OCR Processing**: Multiple preprocessing methods (Adaptive, Otsu, Morphology, Combined)
- **✂️ Region Selection**: Select specific areas of interest for focused text extraction
- **🔄 Auto Deskewing**: Automatically detects and corrects image rotation
- **🎯 Multi-Mode OCR**: Tests 6 different PSM modes and selects the best result
- **🎨 Visual Feedback**: Overlay preview showing detected text with bounding boxes
- **💾 Export Options**: Save extracted text to files or download directly

### Technologies Used

- **Python 3.x**
- **PyQt5** - Desktop GUI framework
- **Streamlit** - Web application framework
- **OpenCV** - Image processing and computer vision
- **PyTesseract** - OCR engine wrapper
- **Tesseract OCR** - Google's OCR engine
- **NumPy** - Numerical operations
- **Pillow (PIL)** - Image manipulation

### Use Cases

- Digitizing printed documents
- Extracting text from photos
- Converting images to editable text
- Document scanning and archiving
- Accessibility tools for text extraction
- Educational projects on OCR technology

### Quick Start

**Desktop App:**
```bash
pip install PyQt5 opencv-python pytesseract numpy
python gui_app.py
```

**Web App:**
```bash
pip install -r requirements_web.txt
streamlit run web_app.py
```

### Project Status

✅ Fully functional with enhanced OCR accuracy
✅ Both desktop and web interfaces working
✅ All assignment requirements met
✅ Production-ready codebase

---

## GitHub Topics/Tags (for repository topics)
```
ocr, pytesseract, computer-vision, image-processing, text-extraction, pyqt5, streamlit, opencv, python, gui-application, web-app, document-scanner, text-recognition, image-to-text, optical-character-recognition
```

