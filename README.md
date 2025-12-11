# 📄 Printed Text Scanner GUI

A comprehensive OCR (Optical Character Recognition) application with both **desktop (PyQt5)** and **web (Streamlit)** interfaces. Extract text from images using advanced preprocessing, multi-mode OCR, auto deskewing, and real-time camera support.

## 🎯 What This Project Does

This project provides a complete solution for extracting printed text from images using computer vision and OCR technology. It offers two interfaces:

- **🖥️ Desktop Application**: Full-featured PyQt5 GUI with live camera feed and drag-and-drop ROI selection
- **🌐 Web Application**: Browser-based Streamlit app accessible from any device

Both versions include advanced image preprocessing, multiple OCR strategies, and visual feedback with text overlay previews.

## Features

- 📷 **Live Camera Input** - Real-time camera feed with frame capture
- 🖼️ **Image Loading** - Load images from your computer
- ✂️ **ROI Selection** - Select Region of Interest (ROI) by clicking and dragging
- 🔍 **OCR Processing** - Extract text from images with preprocessing
- 📝 **Text Display** - View extracted text in the GUI
- 🎨 **Overlay Preview** - See detected text bounding boxes overlaid on the image
- 💾 **Save Functionality** - Save extracted text to files

## Requirements

### Python Packages
```bash
pip install PyQt5 opencv-python pytesseract numpy
```

### Tesseract OCR

**Windows:**
1. Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to default location: `C:\Program Files\Tesseract-OCR`
3. The application will auto-detect the installation

**Alternative Windows Installation:**
- If installed to a different location, edit `gui_app.py` and add your path to the `common_paths` list in the `setup_tesseract_path()` function

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

## How to Run

### Desktop Application (PyQt5)

1. Ensure all dependencies are installed (see Requirements above)
2. Run the application:
```bash
python gui_app.py
```

### Web Application (Streamlit)

1. Install web app dependencies:
```bash
pip install -r requirements_web.txt
```

2. Run the Streamlit web app:
```bash
streamlit run web_app.py
```

3. The app will open in your default web browser at `http://localhost:8501`

**Web App Features:**
- 📤 Image upload from your computer
- 📷 Webcam capture (uses device camera)
- ✂️ ROI selection with coordinate input
- 🔍 Enhanced OCR with multiple preprocessing methods
- 📝 Real-time text extraction and display
- 🎨 Visual overlay showing detected text
- 💾 Download extracted text as .txt file

## Usage Instructions

1. **Start Camera**: Click "Start Camera" to begin live video feed
2. **Capture Frame**: Click "Capture Frame" to freeze the current frame
   - OR click "Load Image" to load an image from your computer
3. **Select ROI (Optional)**: Click "Select ROI" then click and drag on the image to select a region
4. **Run OCR**: Click "Run OCR" to extract text from the image/ROI
   - The extracted text will appear in the text box below
   - Bounding boxes will be overlaid on the image showing detected words
5. **Save Text**: Click "Save Text" to save the extracted text to a file
6. **Clear**: Click "Clear" to reset and start over

## Keyboard Shortcuts

- **Q** - Quit (when camera is running)

## Output

Extracted text files are saved in the `scanned_texts/` directory with timestamps:
- Format: `scanned_text_YYYYMMDD-HHMMSS.txt`

## Troubleshooting

### Tesseract Not Found Error

If you see "TesseractNotFoundError":
1. Ensure Tesseract OCR is installed (see Requirements)
2. On Windows, install to the default location or add it to your PATH
3. Restart the application after installation

### Camera Not Working

- Ensure your camera is connected and not being used by another application
- Try different camera indices if multiple cameras are available

## Project Structure

```
Printed Text Scanner GUI/
├── gui_app.py          # Desktop GUI application (PyQt5)
├── web_app.py          # Web application (Streamlit)
├── requirements_web.txt # Web app dependencies
├── sample1.py          # Face mesh example (reference)
├── sample2.py          # OCR example (reference)
├── scanned_texts/      # Output directory (created automatically)
└── README.md           # This file
```

## Assignment Requirements

This project fulfills the requirements for Week 13 - Assignment 2:
- ✅ GUI interface (PyQt5 Desktop + Streamlit Web)
- ✅ Load or capture image functionality
- ✅ OCR button
- ✅ Display extracted text
- ✅ ROI selection
- ✅ Live camera input (Desktop) / Webcam capture (Web)
- ✅ Overlay preview showing detected text

### Enhanced Features

- 🚀 **Multiple Preprocessing Methods**: Combined, Adaptive, Otsu, Morphology
- 🔄 **Auto Deskewing**: Automatic rotation correction
- 🎯 **Multi-Mode OCR**: Tests multiple PSM modes and selects the best result
- 📊 **Confidence Scoring**: Uses confidence metrics to improve accuracy
- 🧹 **Text Post-Processing**: Cleans and formats extracted text

## License

This project is created for educational purposes.
