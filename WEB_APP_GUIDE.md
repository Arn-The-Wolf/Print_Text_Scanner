# Web Application Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_web.txt
```

Or install individually:
```bash
pip install streamlit opencv-python pytesseract numpy Pillow
```

### 2. Run the Web App

```bash
streamlit run web_app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### 3. Using the Web App

#### Image Input Methods

**Option 1: Upload Image**
1. Select "Upload Image" in the sidebar
2. Click "Choose an image file"
3. Select an image from your computer (PNG, JPG, JPEG, BMP, TIF)

**Option 2: Webcam Capture**
1. Select "Webcam Capture" in the sidebar
2. Allow camera permissions when prompted
3. Click the camera button to capture a photo

#### ROI Selection (Optional)

1. Check "Enable ROI Selection" checkbox
2. Enter coordinates manually:
   - **X1, Y1**: Top-left corner of the region
   - **X2, Y2**: Bottom-right corner of the region
3. The selected region will be highlighted in yellow on the image

**Tip**: You can check the image dimensions by hovering over it, or use the full image by leaving coordinates at default values.

#### Running OCR

1. Configure settings in the sidebar:
   - **Preprocessing Method**: Choose the best method for your image
     - "Combined (Best)" - Recommended for most cases
     - "Adaptive" - Good for varying lighting
     - "Otsu" - Good for consistent lighting
     - "Morphology" - Good for noisy images
   - **Auto Deskew**: Enable to automatically correct image rotation

2. Click the **"🔍 Run OCR"** button

3. Wait for processing (you'll see a spinner)

4. Results will appear:
   - **OCR Overlay Preview**: Image with green bounding boxes showing detected text
   - **Extracted Text**: The recognized text in the right panel

#### Saving Results

1. **Download Text**: Click "💾 Download Text" to download as .txt file
2. **Save to File**: Click "💾 Save to File" to save in the `scanned_texts/` directory

## Features

### ✅ All Desktop App Features

- ✅ Image upload and webcam capture
- ✅ ROI selection (via coordinate input)
- ✅ Multiple preprocessing methods
- ✅ Auto deskewing
- ✅ Multi-mode OCR (tests 6 different PSM modes)
- ✅ Text overlay visualization
- ✅ Text extraction and display
- ✅ Download and save functionality

### 🌐 Web-Specific Advantages

- Access from any device with a browser
- No installation needed (just run the server)
- Shareable via network (if server is accessible)
- Mobile-friendly interface
- Easy to deploy to cloud platforms

## Troubleshooting

### Tesseract Not Found

If you see "TesseractNotFoundError":
1. Install Tesseract OCR (see main README)
2. On Windows, ensure it's in the default location
3. Restart the Streamlit app

### Webcam Not Working

- Ensure camera permissions are granted in your browser
- Try a different browser (Chrome, Firefox, Edge)
- Check if camera is being used by another application

### ROI Selection Issues

- Ensure X2 > X1 and Y2 > Y1
- Coordinates must be within image dimensions
- Use the full image by leaving coordinates at default if unsure

### Performance

- Large images may take longer to process
- The app processes images in real-time
- For best results, use images with clear, high-contrast text

## Deployment

### Local Network Access

To access from other devices on your network:

```bash
streamlit run web_app.py --server.address 0.0.0.0
```

Then access via: `http://YOUR_IP_ADDRESS:8501`

### Cloud Deployment

The app can be deployed to:
- **Streamlit Cloud**: https://streamlit.io/cloud
- **Heroku**: With proper Procfile
- **AWS/Azure/GCP**: Using container services
- **Docker**: Create a Dockerfile with all dependencies

## Browser Compatibility

- ✅ Chrome/Chromium (Recommended)
- ✅ Firefox
- ✅ Edge
- ✅ Safari (may have limited webcam support)
- ⚠️ Mobile browsers (basic functionality)

## Next Steps

1. Try different preprocessing methods to find what works best for your images
2. Experiment with ROI selection for better accuracy on specific regions
3. Use auto deskew for rotated documents
4. Compare results between desktop and web versions

