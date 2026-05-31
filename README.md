# 🌿 AgroVision — SSL & Graph-Refined Object Detection for Precision Agriculture

> **A Streamlit-powered AI web application for real-time crop detection, health scoring, and explainability analysis — built as part of a thesis project on precision agriculture.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo & Screenshots](#demo--screenshots)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Support](#model-support)
- [Visualizations Explained](#visualizations-explained)
- [Detection Pipeline](#detection-pipeline)
- [Export & Reporting](#export--reporting)
- [Configuration & Filters](#configuration--filters)
- [Insights Engine](#insights-engine)
- [System Requirements](#system-requirements)
- [Known Limitations](#known-limitations)
- [Author](#author)

---

## Overview

**AgroVision** (v2.1) is an interactive precision agriculture AI platform designed to assist farmers, researchers, and agronomists in analyzing crop images with deep learning. The application accepts user-uploaded YOLO `.pt` or scikit-learn `.pkl` models and runs object detection on one or multiple crop images (currently supporting **Sunflower** and **Rice** crops).

The system provides a comprehensive analysis pipeline beyond simple bounding-box detection — including Grad-CAM explainability heatmaps, density maps, proximity analysis, 3D detection visualization, image quality assessment, statistical analytics, and downloadable PDF reports.

This project was developed as a thesis application exploring **Self-Supervised Learning (SSL)** and **graph-refinement techniques** for agricultural object detection.

---

## Features

### 🔍 Detection
- Real-time YOLOv8 object detection on uploaded crop images
- Support for multiple simultaneous image uploads (batch mode)
- Advanced bounding box rendering with corner accents and class color-coding
- Optional grid overlay for spatial reference
- Per-detection metadata: class, confidence, coordinates, area, aspect ratio

### 🔥 Explainability (Grad-CAM)
- **Grad-CAM**: Broad Gaussian blob-based attention maps (sigma=0.9) showing model focus areas
- **Grad-CAM++**: Tighter, confidence²-weighted heatmaps (sigma=0.35) for sharper localisation
- Both overlay and raw grayscale heatmap views side-by-side

### 🌡️ Heatmaps
- **Density Heatmap**: Turbo colormap overlay showing detection confidence distribution across the image
- **Proximity Map**: Viridis colormap visualization of inter-detection distances

### 📊 Analytics & Charts
- Class distribution pie chart
- Confidence score histogram (with mean/median markers)
- Confidence vs. bounding box size scatter plot
- Bounding box width/height distributions
- Aspect ratio distribution histogram
- 5×5 spatial detection density grid
- ROC curve analysis (AUC displayed)
- 3D interactive scatter plot (Plotly) — position × confidence

### 🔬 Image Quality Analysis
- Laplacian variance-based blur/sharpness score
- Brightness assessment
- Canny edge detection visualization
- LAB-space histogram equalization
- RGB color channel distribution chart

### 📈 Health Scoring
- Composite crop health score (0–100) based on:
  - Detected area coverage (30%)
  - Average detection confidence (40%)
  - Detection count relative to image size (30%)

### 💾 Export
- Annotated image download (PNG)
- Detection results as CSV
- Detection data as structured JSON
- Per-section PDF reports with AI insights
- Full analysis PDF report
- Batch ZIP export for multiple images

---

## Demo & Screenshots

| Grad-CAM Heatmap on Rice | Grad-CAM Heatmap on Sunflowers |
|:---:|:---:|
| ![Rice Heatmap]([https://github.com/karibshams/thesis_app/blob/main/rice_heatmap.jpeg]) | ![Sunflower Heatmap]([https://github.com/karibshams/thesis_app/blob/main/sunflower_heatmap.jpeg]) |

> The JET colormap overlays show where the model attends — **red/yellow = high activation**, **blue = background**.

---

## Tech Stack

| Category | Library / Version |
|---|---|
| Web Framework | `streamlit==1.28.1` |
| Object Detection | `ultralytics==8.0.203` (YOLOv8) |
| Deep Learning | `torch==2.5.1`, `torchvision==0.20.1` |
| Computer Vision | `opencv-python==4.8.1.78` |
| Image Processing | `pillow==10.0.1`, `scikit-image==0.22.0` |
| Data Analysis | `numpy==1.26.4`, `pandas==2.1.4`, `scipy==1.11.4` |
| Visualization | `matplotlib==3.8.2`, `plotly==5.18.0` |
| PDF Generation | `reportlab==4.0.9` |

---

## Project Structure

```
thesis_app/
│
├── agrovision.py          # Main Streamlit application (all logic & UI)
├── app.py                 # App entry point / runner
├── requirements.txt       # Python dependencies
├── temp_model.pt          # Placeholder/default YOLO model weights
├── README.md              # Project documentation
└── .gitignore
```

> **Note:** The core application logic, UI, detection pipeline, visualization engine, and PDF generator are all contained within `agrovision.py`.

---

## Installation

### Prerequisites
- Python 3.9 or higher
- pip
- (Optional but recommended) a CUDA-capable GPU for faster inference

### Step 1 — Clone the Repository

```bash
git clone https://github.com/karibshams/thesis_app.git
cd thesis_app
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **PyTorch Note:** The `requirements.txt` installs the default CPU/CUDA build of PyTorch. For a specific CUDA version, install PyTorch manually from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) before running the above.

### Step 4 — Run the App

```bash
streamlit run agrovision.py
```

The app will open at `http://localhost:8501` in your browser.

---

## Usage

### 1. Upload a Model
- In the **left sidebar**, under **Upload Models**, upload your trained `.pt` (YOLO) or `.pkl` (scikit-learn) model file.
- Up to **10 models** can be uploaded simultaneously.
- Select the **Active Model** from the dropdown.

### 2. Configure Detection Settings
Use the sidebar to adjust:
- **Confidence Threshold** (default: 0.50) — minimum score to accept a detection
- **IOU Threshold** (default: 0.45) — for non-max suppression overlap
- **Visualization toggles** — enable/disable Grad-CAM, Density Map, 3D View, etc.

### 3. Upload Crop Images
- Click **Upload Crop Images** in the main area.
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Multiple images can be uploaded at once for batch analysis.

### 4. Explore Results
Each image gets a full expandable analysis panel with **8 tabs**:

| Tab | Contents |
|---|---|
| 📸 Detection | Original vs. annotated image side-by-side |
| 🔥 Grad-CAM | Grad-CAM & Grad-CAM++ overlays + raw heatmaps |
| 🌡️ Heatmaps | Density heatmap + proximity map |
| 3️⃣ 3D View | Interactive Plotly 3D scatter of detections |
| 🔬 Image Analysis | Quality metrics, edge detection, color histograms |
| 📊 Analytics | Class distribution, confidence distribution |
| 📈 Advanced Charts | Scatter, aspect ratio, density grid, ROC curve |
| 📋 Details | Full detection table, class-wise statistics |

### 5. Apply Filters
In the sidebar under **Detection Filters**:
- Filter detections by **class** (All / Sunflower / Rice)
- Set minimum/maximum **confidence** range
- Set minimum/maximum **bounding box area** (px²)

### 6. Export Results
At the bottom of each image panel:
- Download the **annotated image** (PNG)
- Download **detections as CSV**
- Download **detections as JSON**
- Generate and download a **section-specific PDF** or **full PDF report**

For multiple images, use the **Batch Export → Download All as ZIP** button.

---

## Model Support

### YOLOv8 `.pt` Models
The app is built around Ultralytics YOLOv8 models. Any `.pt` file exported from YOLOv8 training will work. The model is loaded via:

```python
from ultralytics import YOLO
model = YOLO("your_model.pt")
```

Detection is run at `imgsz=640` by default.

Expected classes in the model for full feature support: `Sunflower`, `Rice` (class names must match for color-coding and filters to apply correctly).

### scikit-learn `.pkl` Models
PKL models can be uploaded and loaded. The app currently displays the model type and leaves the prediction integration point open for custom implementation. Extend the `elif model_type == "pkl":` block in the main app section to add custom prediction logic.

---

## Visualizations Explained

### Grad-CAM vs. Grad-CAM++

Both are implemented as **detection-guided Gaussian heatmaps** (not gradient backpropagation through the model, since YOLO is used as a black-box detector):

| | Grad-CAM | Grad-CAM++ |
|---|---|---|
| **Sigma scale** | 0.9 (wide blobs) | 0.35 (tight blobs) |
| **Weighting** | `confidence` | `confidence²` |
| **Effect** | Broad contextual attention | Sharp, localised peaks |
| **Colormap** | JET (blue→red) | JET (blue→red) |

Both are blended 50/50 with the original image.

### Density Heatmap
Each detection box fills a region of the heatmap weighted by its confidence score. The result is normalized and rendered with the **TURBO** colormap.

### Proximity Map
Draws lines between all detection center pairs, with intensity proportional to the Euclidean distance between them. Rendered with the **VIRIDIS** colormap.

### 3D Detection Plot
Each detection is plotted at `(CenterX, CenterY, Confidence)` in a Plotly 3D scatter chart. Bubble size encodes bounding box area.

---

## Detection Pipeline

```
Input Image (PIL)
       │
       ▼
  pil_to_cv2()              # Convert to BGR numpy array
       │
       ▼
  run_detection()           # YOLOv8 inference at conf + IOU thresholds
       │
       ▼
  draw_detections_advanced() # Draw boxes, corner accents, labels, center dots
       │
       ▼
  filter_detections()       # Apply sidebar class/conf/size filters
       │
       ├──► analyze_image_quality()      → blur, contrast, brightness
       ├──► calculate_health_score()     → composite 0–100 score
       ├──► create_class_statistics()    → per-class summary DataFrame
       │
       ├──► generate_gradcam()           → Grad-CAM overlay
       ├──► generate_gradcam_plus_plus() → Grad-CAM++ overlay
       ├──► generate_density_heatmap()   → Turbo density map
       ├──► create_proximity_map()       → Viridis proximity map
       ├──► create_3d_plot()             → Plotly 3D scatter
       │
       └──► Export: PNG / CSV / JSON / PDF / ZIP
```

---

## Export & Reporting

### Per-Section PDF
Each analysis tab has a **📄 PDF** button. Clicking it generates a PDF containing:
- AgroVision header + timestamp
- Key metrics table (health score, detections, confidence, image quality)
- The relevant visualization figure
- AI analysis insight text (technical + plain language)

### Full PDF Report
The **📑 Full PDF Report** button generates a comprehensive document with:
- Header, subtitle, and generation timestamp
- Full metrics summary table
- Original image vs. annotated image (side-by-side)
- Detection insight paragraph
- Class-wise statistics table
- Complete detection records table

### JSON Export Schema
```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "image_filename": "crop_image.jpg",
  "total_detections": 12,
  "detections": [
    {
      "Class": "Sunflower",
      "Confidence": "0.9234",
      "X1": 120, "Y1": 85, "X2": 310, "Y2": 280,
      "CenterX": 215, "CenterY": 182,
      "Width": 190, "Height": 195,
      "Area": 37050,
      "AspectRatio": "0.97"
    }
  ]
}
```

---

## Configuration & Filters

| Parameter | Default | Range | Description |
|---|---|---|---|
| Confidence Threshold | 0.50 | 0.10–1.00 | Minimum detection confidence |
| IOU Threshold | 0.45 | 0.10–1.00 | Non-max suppression overlap |
| Min Confidence Filter | 0.00 | 0.00–1.00 | Post-detection display filter |
| Max Confidence Filter | 1.00 | 0.00–1.00 | Post-detection display filter |
| Min Area (px²) | 0 | 0–∞ | Filter tiny detections |
| Max Area (px²) | 10,000,000 | 0–∞ | Filter oversized detections |
| Class Filter | All | All/Sunflower/Rice | Show only selected classes |

---

## Insights Engine

Every visualization section generates two levels of AI insight:

- **Technical**: Quantitative analysis with statistical values (σ, mean, AUC, etc.)
- **Plain Language**: A human-readable explanation suitable for non-expert users (farmers, extension workers)

These insights are rendered inline in the app and embedded into exported PDFs.

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.10+ |
| RAM | 4 GB | 8 GB+ |
| GPU | Not required | CUDA 11.8+ (for faster inference) |
| Storage | ~2 GB | ~4 GB (for model weights + dependencies) |
| OS | Windows 10 / Ubuntu 20.04 / macOS 11 | Any modern 64-bit OS |

---

## Known Limitations

- **Grad-CAM implementation** uses Gaussian approximation from detected box centers, not true gradient backpropagation through YOLO layers. This is intentional for speed and YOLO compatibility, but differs from academic Grad-CAM on classification CNNs.
- **ROC curve** is approximated from detection confidence scores against thresholds — not computed from a labeled ground-truth test set.
- **PKL model integration** is scaffolded but requires custom prediction logic to be added by the user.
- Batch ZIP export re-runs inference on all uploaded images, which may be slow for large batches without a GPU.
- The app currently targets **Sunflower** and **Rice** crop classes. Other classes will be detected if present in the model but won't receive class-specific color coding.

---

## Author

**Karib Shams**
- GitHub: [@karibshams](https://github.com/karibshams)
- Repository: [github.com/karibshams/thesis_app](https://github.com/karibshams/thesis_app)

---

<div align="center">
  <sub>🌿 AgroVision · SSL & Graph-Refined Object Detection · Precision Agriculture AI</sub>
</div>
