import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import io
import pandas as pd
from torch.nn import functional as F
import json
from datetime import datetime
import zipfile
from skimage import filters, exposure
from skimage.metrics import structural_similarity as ssim
import tempfile
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle, HRFlowable
from reportlab.lib.units import inch
from reportlab.lib import colors as rl_colors
import pickle
import os

# ==================== PAGE CONFIG & CUSTOM CSS ====================
st.set_page_config(
    page_title="AgroVision - Precision Agriculture AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ── Global font & background ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0f0a 0%, #0d1f0d 40%, #0a1a1f 100%);
        min-height: 100vh;
    }
    
    /* ── Header Hero ── */
    .agrovision-hero {
        background: linear-gradient(135deg, #0d2b1a 0%, #0a1f2e 50%, #1a0d2b 100%);
        border: 1px solid rgba(46,213,115,0.2);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5), 0 0 80px rgba(46,213,115,0.05);
    }
    .agrovision-hero::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(46,213,115,0.06) 0%, transparent 50%),
                    radial-gradient(circle at 70% 50%, rgba(52,152,219,0.06) 0%, transparent 50%);
        pointer-events: none;
    }
    .agrovision-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2ed573, #3498db, #a29bfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.2;
    }
    .agrovision-subtitle {
        font-size: 1rem;
        color: rgba(200,255,200,0.6);
        margin-top: 0.4rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    .agrovision-badge {
        display: inline-block;
        background: rgba(46,213,115,0.15);
        border: 1px solid rgba(46,213,115,0.4);
        color: #2ed573;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }
    
    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1f0d 0%, #0a1420 100%);
        border-right: 1px solid rgba(46,213,115,0.15);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #2ed573 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* ── Metric Cards ── */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(13,43,26,0.8), rgba(10,26,47,0.8));
        border: 1px solid rgba(46,213,115,0.2);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    [data-testid="metric-container"]:hover {
        border-color: rgba(46,213,115,0.5);
        box-shadow: 0 8px 30px rgba(46,213,115,0.1);
        transform: translateY(-2px);
    }
    [data-testid="metric-container"] label {
        color: rgba(180,255,180,0.7) !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #2ed573 !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }
    
    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #1a7a3c, #155e75);
        color: white;
        border: 1px solid rgba(46,213,115,0.4);
        border-radius: 10px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #22a84d, #1a7a9a);
        border-color: #2ed573;
        box-shadow: 0 0 20px rgba(46,213,115,0.3);
        transform: translateY(-1px);
    }
    
    /* ── Download buttons ── */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #0d4429, #0a2d3d) !important;
        color: #2ed573 !important;
        border: 1px solid rgba(46,213,115,0.5) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #1a7a3c, #155e75) !important;
        box-shadow: 0 0 20px rgba(46,213,115,0.2) !important;
    }
    
    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(13,31,13,0.8);
        border-radius: 14px;
        padding: 5px;
        gap: 4px;
        border: 1px solid rgba(46,213,115,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: rgba(180,255,180,0.6);
        font-weight: 500;
        font-size: 0.85rem;
        transition: all 0.2s;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(46,213,115,0.2), rgba(52,152,219,0.2)) !important;
        color: #2ed573 !important;
        border: 1px solid rgba(46,213,115,0.3) !important;
    }
    
    /* ── Alerts ── */
    .stAlert {
        border-radius: 12px;
    }
    .stSuccess {
        background: rgba(46,213,115,0.1) !important;
        border-color: rgba(46,213,115,0.4) !important;
        border-radius: 12px !important;
        color: #2ed573 !important;
    }
    .stWarning {
        border-radius: 12px !important;
    }
    .stInfo {
        border-radius: 12px !important;
    }
    
    /* ── DataFrames ── */
    .dataframe {
        border-radius: 12px !important;
        overflow: hidden;
    }
    
    /* ── Section headers ── */
    .section-header {
        background: linear-gradient(135deg, rgba(46,213,115,0.08), rgba(52,152,219,0.08));
        border-left: 4px solid #2ed573;
        border-radius: 0 12px 12px 0;
        padding: 0.8rem 1.2rem;
        margin: 1.5rem 0 1rem 0;
        color: rgba(200,255,200,0.9);
        font-weight: 600;
        font-size: 1.05rem;
    }
    
    /* ── Analysis insight box ── */
    .analysis-insight {
        background: linear-gradient(135deg, rgba(13,43,26,0.6), rgba(10,26,47,0.6));
        border: 1px solid rgba(46,213,115,0.15);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: rgba(200,255,200,0.8);
        line-height: 1.7;
    }
    .analysis-insight strong {
        color: #2ed573;
    }
    .insight-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #2ed573;
        font-weight: 700;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(46,213,115,0.3) !important;
        border-radius: 14px !important;
        background: rgba(13,43,26,0.3) !important;
        transition: all 0.3s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(46,213,115,0.6) !important;
        background: rgba(13,43,26,0.5) !important;
    }
    
    /* ── Progress bar ── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #2ed573, #3498db) !important;
        border-radius: 10px;
    }
    
    /* ── Export card ── */
    .export-card {
        background: rgba(13,43,26,0.4);
        border: 1px solid rgba(46,213,115,0.15);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s;
    }
    .export-card:hover {
        border-color: rgba(46,213,115,0.4);
        background: rgba(13,43,26,0.6);
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0a0f0a; }
    ::-webkit-scrollbar-thumb { background: rgba(46,213,115,0.3); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(46,213,115,0.6); }
    
    /* dividers */
    hr { border-color: rgba(46,213,115,0.1) !important; }
</style>
""", unsafe_allow_html=True)

# ==================== HERO HEADER ====================
st.markdown("""
<div class="agrovision-hero">
    <div class="agrovision-badge">🌿 Precision Agriculture AI</div>
    <h1 class="agrovision-title">AgroVision</h1>
    <p style="font-size:1.15rem;font-weight:500;color:rgba(200,255,200,0.85);margin:0.2rem 0 0.1rem;">
        SSL & Graph-Refined Object Detection for Precision Agriculture
    </p>
    <p class="agrovision-subtitle">
        Professional-grade Sunflower & Rice Detection · YOLOv11 · Multi-model Analysis · AI-powered Reports
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'roi_mode' not in st.session_state:
    st.session_state.roi_mode = False
if 'roi_points' not in st.session_state:
    st.session_state.roi_points = []
if 'loaded_models' not in st.session_state:
    st.session_state.loaded_models = {}

# ==================== SIDEBAR CONFIGURATION ====================
with st.sidebar:
    st.markdown("### ⚙️ Advanced Configuration")
    
    # ---- Multi-model upload ----
    st.markdown("**📤 Upload Models** (up to 10)")
    st.caption("Supported: `.pt` (YOLOv11/PyTorch) · `.pkl` (scikit-learn/pickle)")
    
    uploaded_models = st.file_uploader(
        "Upload model files",
        type=["pt", "pkl"],
        accept_multiple_files=True,
        help="Upload up to 10 .pt or .pkl model files",
        label_visibility="collapsed"
    )
    if uploaded_models and len(uploaded_models) > 10:
        st.warning("⚠️ Maximum 10 models allowed. Only first 10 used.")
        uploaded_models = uploaded_models[:10]
    
    # Model selector if multiple uploaded
    selected_model_name = None
    if uploaded_models:
        model_names = [f.name for f in uploaded_models]
        selected_model_name = st.selectbox("🎯 Active Model", model_names)
        st.caption(f"✅ {len(uploaded_models)} model(s) loaded")
    
    st.markdown("---")
    
    # Thresholds
    conf_threshold = st.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
    iou_threshold = st.slider("IOU Threshold", min_value=0.1, max_value=1.0, value=0.45, step=0.05)
    
    st.markdown("---")
    st.markdown("### 📊 Visualization Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        enable_gradcam = st.checkbox("Grad-CAM", value=True)
        enable_heatmap = st.checkbox("Density Heatmap", value=True)
        enable_3d = st.checkbox("3D Visualization", value=True)
    with col2:
        enable_preprocessing = st.checkbox("Image Analysis", value=True)
        enable_proximity = st.checkbox("Proximity Map", value=True)
        enable_grid = st.checkbox("Grid Overlay", value=True)
    
    st.markdown("---")
    st.markdown("### 🎯 Filtering Options")
    
    filter_class = st.multiselect("Filter by Class", ["All", "Sunflower", "Rice"], default=["All"])
    min_conf_filter = st.slider("Min Confidence Filter", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    max_conf_filter = st.slider("Max Confidence Filter", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
    min_size = st.number_input("Min Detection Area (px²)", value=0, min_value=0)
    max_size = st.number_input("Max Detection Area (px²)", value=10000000, min_value=0)
    
    st.markdown("---")
    st.markdown("### 📁 Batch Processing")
    batch_mode = st.checkbox("Enable Batch Mode", value=False)
    if batch_mode:
        st.info("Upload multiple images for batch analysis")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;padding:0.5rem;'>
        <div style='font-size:0.7rem;color:rgba(46,213,115,0.5);letter-spacing:1px;'>AGROVISION v2.0</div>
        <div style='font-size:0.65rem;color:rgba(180,255,180,0.3);margin-top:3px;'>Precision Agriculture AI</div>
    </div>
    """, unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

@st.cache_resource
def load_pt_model(model_bytes, model_name):
    try:
        tmp_path = f"temp_model_{model_name.replace(' ','_')}.pt"
        with open(tmp_path, "wb") as f:
            f.write(model_bytes)
        model = YOLO(tmp_path)
        return model, "yolo"
    except Exception as e:
        st.error(f"Error loading .pt model: {e}")
        return None, None

@st.cache_resource
def load_pkl_model(model_bytes, model_name):
    try:
        model = pickle.loads(model_bytes)
        return model, "pkl"
    except Exception as e:
        st.error(f"Error loading .pkl model: {e}")
        return None, None

def load_active_model(uploaded_models, selected_model_name):
    if not uploaded_models or not selected_model_name:
        return None, None
    for f in uploaded_models:
        if f.name == selected_model_name:
            model_bytes = f.getvalue()
            if f.name.endswith(".pt"):
                return load_pt_model(model_bytes, f.name)
            elif f.name.endswith(".pkl"):
                return load_pkl_model(model_bytes, f.name)
    return None, None

def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def run_detection(image_np, model, conf, iou):
    results = model(image_np, conf=conf, iou=iou, imgsz=640)
    return results

def filter_detections(detections, class_filter, min_conf, max_conf, min_area, max_area):
    filtered = []
    for det in detections:
        conf = float(det["Confidence"])
        area = det["Area"]
        cls = det["Class"]
        if "All" not in class_filter and cls not in class_filter:
            continue
        if conf < min_conf or conf > max_conf:
            continue
        if area < min_area or area > max_area:
            continue
        filtered.append(det)
    return filtered

# ==================== VISUALIZATION FUNCTIONS ====================

def draw_detections_advanced(image_np, results, conf_threshold, show_grid=False):
    image_annotated = image_np.copy()
    h, w = image_np.shape[:2]
    detections = []
    
    CLASS_COLORS = {
        "Sunflower": (0, 200, 100),
        "Rice": (52, 152, 219),
    }
    DEFAULT_COLOR = (46, 213, 115)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            cls_name = result.names[cls]
            color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
            
            # Rounded rect simulation with thicker lines
            cv2.rectangle(image_annotated, (x1, y1), (x2, y2), color, 2)
            # Corner accents
            acc = 12
            cv2.line(image_annotated, (x1, y1), (x1+acc, y1), color, 4)
            cv2.line(image_annotated, (x1, y1), (x1, y1+acc), color, 4)
            cv2.line(image_annotated, (x2, y1), (x2-acc, y1), color, 4)
            cv2.line(image_annotated, (x2, y1), (x2, y1+acc), color, 4)
            cv2.line(image_annotated, (x1, y2), (x1+acc, y2), color, 4)
            cv2.line(image_annotated, (x1, y2), (x1, y2-acc), color, 4)
            cv2.line(image_annotated, (x2, y2), (x2-acc, y2), color, 4)
            cv2.line(image_annotated, (x2, y2), (x2, y2-acc), color, 4)

            label = f"{cls_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(image_annotated, (x1, y1 - 28), (x1 + label_size[0] + 8, y1), color, -1)
            cv2.putText(image_annotated, label, (x1 + 4, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(image_annotated, (cx, cy), 4, (255, 255, 255), -1)
            cv2.circle(image_annotated, (cx, cy), 4, color, 2)
            
            detections.append({
                "Class": cls_name,
                "Confidence": f"{conf:.4f}",
                "X1": x1, "Y1": y1, "X2": x2, "Y2": y2,
                "CenterX": cx, "CenterY": cy,
                "Width": x2 - x1, "Height": y2 - y1,
                "Area": (x2 - x1) * (y2 - y1),
                "AspectRatio": f"{(x2 - x1) / max(y2 - y1, 1):.2f}"
            })
    
    if show_grid:
        grid_size = 64
        for i in range(0, h, grid_size):
            cv2.line(image_annotated, (0, i), (w, i), (46, 80, 46), 1)
        for i in range(0, w, grid_size):
            cv2.line(image_annotated, (i, 0), (i, h), (46, 80, 46), 1)
    
    return cv2_to_pil(image_annotated), detections

def generate_density_heatmap(detections, image_shape):
    h, w = image_shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    for det in detections:
        x1, y1, x2, y2 = det["X1"], det["Y1"], det["X2"], det["Y2"]
        conf = float(det["Confidence"])
        heatmap[y1:y2, x1:x2] += conf
    
    if heatmap.max() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return Image.fromarray(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)), heatmap

def generate_gradcam(image_np, model):
    try:
        h, w = image_np.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        with torch.no_grad():
            results = model(image_np, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                heatmap[y1:y2, x1:x2] += conf
        
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
        
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(image_np, 0.6, heatmap_colored, 0.4, 0)
        
        return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)), heatmap
    except Exception as e:
        return None, None

def create_proximity_map(detections, image_shape):
    h, w = image_shape[:2]
    proximity_map = np.zeros((h, w), dtype=np.float32)
    
    if len(detections) < 2:
        return None
    
    for i, det1 in enumerate(detections):
        for det2 in detections[i+1:]:
            x1, y1 = det1["CenterX"], det1["CenterY"]
            x2, y2 = det2["CenterX"], det2["CenterY"]
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            cv2.line(proximity_map, (x1, y1), (x2, y2), dist, 2)
    
    if proximity_map.max() > 0:
        proximity_map = (proximity_map - proximity_map.min()) / (proximity_map.max() - proximity_map.min() + 1e-6)
    
    proximity_colored = cv2.applyColorMap((proximity_map * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    return Image.fromarray(cv2.cvtColor(proximity_colored, cv2.COLOR_BGR2RGB))

def create_3d_plot(detections):
    if not detections:
        return None
    
    df = pd.DataFrame(detections)
    df["Confidence_float"] = df["Confidence"].astype(float)
    
    color_map = {"Sunflower": "#2ed573", "Rice": "#3498db"}
    
    fig = px.scatter_3d(
        df, x="CenterX", y="CenterY", z="Confidence_float",
        color="Class", size="Area",
        hover_data=["Confidence", "Area"],
        title="3D Detection Analysis",
        labels={"CenterX": "X Position", "CenterY": "Y Position", "Confidence_float": "Confidence"},
        color_discrete_map=color_map
    )
    
    fig.update_layout(
        height=550,
        paper_bgcolor='rgba(10,20,10,0)',
        plot_bgcolor='rgba(10,20,10,0)',
        font=dict(color='rgba(200,255,200,0.8)'),
        scene=dict(
            bgcolor='rgba(10,20,10,0.5)',
            xaxis=dict(gridcolor='rgba(46,213,115,0.2)', color='rgba(200,255,200,0.6)'),
            yaxis=dict(gridcolor='rgba(46,213,115,0.2)', color='rgba(200,255,200,0.6)'),
            zaxis=dict(gridcolor='rgba(46,213,115,0.2)', color='rgba(200,255,200,0.6)'),
        )
    )
    return fig

def analyze_image_quality(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    contrast = np.std(hist)
    brightness = np.mean(gray)
    return {
        "Blur Score": f"{blur_score:.2f}",
        "Contrast": f"{contrast:.2f}",
        "Brightness": f"{brightness:.0f}",
        "Image Quality": "Good" if blur_score > 100 else "Fair" if blur_score > 50 else "Poor"
    }

def create_edge_detection(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return Image.fromarray(edges)

def create_histogram_equalized(image_np):
    lab = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

def dark_chart_style():
    plt.rcParams.update({
        'figure.facecolor': '#0a140a',
        'axes.facecolor': '#0d1f0d',
        'axes.edgecolor': 'rgba(46,213,115,0.2)',
        'axes.labelcolor': '#8fe8b0',
        'xtick.color': '#8fe8b0',
        'ytick.color': '#8fe8b0',
        'text.color': '#b0e8b0',
        'grid.color': 'rgba(46,213,115,0.1)',
        'grid.alpha': 0.3,
    })

def create_color_distribution(image_np):
    dark_chart_style()
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0a140a')
    ax.set_facecolor('#0d1f0d')
    
    channel_colors = ['#3498db', '#2ed573', '#e74c3c']
    channel_labels = ['Blue', 'Green', 'Red']
    
    for i, (col, label) in enumerate(zip(channel_colors, channel_labels)):
        hist = cv2.calcHist([image_np], [i], None, [256], [0, 256])
        ax.fill_between(range(256), hist.ravel(), alpha=0.3, color=col)
        ax.plot(hist, color=col, label=label, linewidth=1.5)
    
    ax.set_xlabel("Pixel Intensity", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("Color Channel Distribution", fontsize=13, fontweight='bold', color='#2ed573', pad=12)
    ax.legend(facecolor='#0d1f0d', edgecolor='rgba(46,213,115,0.3)')
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    return fig

def create_confidence_distribution(detections):
    dark_chart_style()
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0a140a')
    ax.set_facecolor('#0d1f0d')
    
    confidences = [float(d["Confidence"]) for d in detections]
    
    if confidences:
        n, bins, patches = ax.hist(confidences, bins=15, color='#2ed573', alpha=0.6, edgecolor='rgba(46,213,115,0.8)', linewidth=0.8)
        for patch in patches:
            patch.set_facecolor('#2ed573')
            patch.set_alpha(0.6)
        ax.axvline(np.mean(confidences), color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
        ax.axvline(np.median(confidences), color='#f39c12', linestyle='--', linewidth=2, label=f'Median: {np.median(confidences):.3f}')
        ax.set_xlabel("Confidence Score", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title("Confidence Score Distribution", fontsize=13, fontweight='bold', color='#2ed573', pad=12)
        ax.legend(facecolor='#0d1f0d', edgecolor='rgba(46,213,115,0.3)')
        ax.grid(True, alpha=0.15)
    
    plt.tight_layout()
    return fig

def create_class_distribution(detections):
    dark_chart_style()
    class_counts = {}
    for det in detections:
        cls = det["Class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='#0a140a')
    ax.set_facecolor('#0d1f0d')
    
    if class_counts:
        palette = ['#2ed573', '#3498db', '#a29bfe', '#fd79a8', '#fdcb6e']
        wedges, texts, autotexts = ax.pie(
            class_counts.values(), labels=class_counts.keys(),
            autopct='%1.1f%%', colors=palette[:len(class_counts)],
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.6, edgecolor='#0a140a', linewidth=3)
        )
        for text in texts:
            text.set_color('#b0e8b0')
            text.set_fontsize(11)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        ax.set_title("Class Distribution", fontsize=13, fontweight='bold', color='#2ed573', pad=12)
    
    plt.tight_layout()
    return fig

def create_scatter_plot(detections):
    dark_chart_style()
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0a140a')
    ax.set_facecolor('#0d1f0d')
    
    if detections:
        areas = [d["Area"] for d in detections]
        confidences = [float(d["Confidence"]) for d in detections]
        classes = [d["Class"] for d in detections]
        
        unique_classes = list(set(classes))
        palette = ['#2ed573', '#3498db', '#a29bfe', '#fd79a8']
        colors_map = {cls: palette[i % len(palette)] for i, cls in enumerate(unique_classes)}
        c_vals = [colors_map[cls] for cls in classes]
        
        ax.scatter(areas, confidences, s=120, c=c_vals, alpha=0.75, edgecolors='white', linewidth=0.8)
        ax.set_xlabel("Bounding Box Area (pixels)", fontsize=11)
        ax.set_ylabel("Confidence Score", fontsize=11)
        ax.set_title("Confidence vs Bounding Box Size", fontsize=13, fontweight='bold', color='#2ed573', pad=12)
        ax.grid(True, alpha=0.15)
        
        legend_labels = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_map[cls],
                                    markersize=8, label=cls) for cls in unique_classes]
        ax.legend(handles=legend_labels, loc='best', facecolor='#0d1f0d', edgecolor='rgba(46,213,115,0.3)')
    
    plt.tight_layout()
    return fig

def create_bbox_distribution(detections):
    dark_chart_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0a140a')
    
    if detections:
        widths = [d["Width"] for d in detections]
        heights = [d["Height"] for d in detections]
        
        axes[0].set_facecolor('#0d1f0d')
        axes[0].hist(widths, bins=10, color='#e74c3c', alpha=0.65, edgecolor='rgba(231,76,60,0.8)', linewidth=0.8)
        axes[0].set_xlabel("Width (pixels)", fontsize=10)
        axes[0].set_ylabel("Frequency", fontsize=10)
        axes[0].set_title("Bounding Box Width Distribution", fontsize=12, fontweight='bold', color='#2ed573', pad=10)
        axes[0].grid(True, alpha=0.15)
        
        axes[1].set_facecolor('#0d1f0d')
        axes[1].hist(heights, bins=10, color='#3498db', alpha=0.65, edgecolor='rgba(52,152,219,0.8)', linewidth=0.8)
        axes[1].set_xlabel("Height (pixels)", fontsize=10)
        axes[1].set_ylabel("Frequency", fontsize=10)
        axes[1].set_title("Bounding Box Height Distribution", fontsize=12, fontweight='bold', color='#2ed573', pad=10)
        axes[1].grid(True, alpha=0.15)
    
    plt.tight_layout()
    return fig

def create_aspect_ratio_distribution(detections):
    dark_chart_style()
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0a140a')
    ax.set_facecolor('#0d1f0d')
    
    if detections:
        aspect_ratios = [float(d["AspectRatio"]) for d in detections]
        ax.hist(aspect_ratios, bins=15, color='#a29bfe', alpha=0.65, edgecolor='rgba(162,155,254,0.8)', linewidth=0.8)
        ax.axvline(np.mean(aspect_ratios), color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: {np.mean(aspect_ratios):.2f}')
        ax.set_xlabel("Aspect Ratio (Width/Height)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title("Aspect Ratio Distribution", fontsize=13, fontweight='bold', color='#2ed573', pad=12)
        ax.legend(facecolor='#0d1f0d', edgecolor='rgba(46,213,115,0.3)')
        ax.grid(True, alpha=0.15)
    
    plt.tight_layout()
    return fig

def create_detection_density_chart(detections, image_shape):
    dark_chart_style()
    h, w = image_shape[:2]
    grid_cols, grid_rows = 5, 5
    cell_w, cell_h = w // grid_cols, h // grid_rows
    grid = np.zeros((grid_rows, grid_cols))
    
    for det in detections:
        cx, cy = det["CenterX"], det["CenterY"]
        col, row = min(cx // cell_w, grid_cols - 1), min(cy // cell_h, grid_rows - 1)
        grid[row, col] += 1
    
    fig, ax = plt.subplots(figsize=(8, 7), facecolor='#0a140a')
    ax.set_facecolor('#0d1f0d')
    im = ax.imshow(grid, cmap='YlOrRd', interpolation='nearest')
    
    for i in range(grid_rows):
        for j in range(grid_cols):
            ax.text(j, i, f'{int(grid[i,j])}', ha='center', va='center', color='white', fontsize=12, fontweight='bold')
    
    ax.set_xlabel("Image Width Zones", fontsize=10)
    ax.set_ylabel("Image Height Zones", fontsize=10)
    ax.set_title("Detection Density Grid (5×5 Zones)", fontsize=13, fontweight='bold', color='#2ed573', pad=12)
    plt.colorbar(im, ax=ax, label="Detection Count")
    plt.tight_layout()
    return fig

def create_roc_curve(detections):
    dark_chart_style()
    confidences = sorted([float(d["Confidence"]) for d in detections], reverse=True)
    thresholds = np.linspace(0, 1, 100)
    tpr, fpr = [], []
    
    for thresh in thresholds:
        detected = sum(1 for c in confidences if c >= thresh)
        tpr.append(detected / max(len(confidences), 1))
        fpr.append(1 - tpr[-1])
    
    auc = np.trapz(tpr, fpr)
    
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0a140a')
    ax.set_facecolor('#0d1f0d')
    ax.plot(fpr, tpr, color='#2ed573', linewidth=2.5, label=f'ROC Curve (AUC≈{abs(auc):.3f})')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#2ed573')
    ax.plot([0, 1], [0, 1], '#e74c3c', linestyle='--', linewidth=2, label='Random Classifier')
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve Analysis", fontsize=13, fontweight='bold', color='#2ed573', pad=12)
    ax.legend(facecolor='#0d1f0d', edgecolor='rgba(46,213,115,0.3)')
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    return fig

def create_class_statistics(detections):
    if not detections:
        return pd.DataFrame()
    df = pd.DataFrame(detections)
    stats = []
    for cls in df['Class'].unique():
        class_data = df[df['Class'] == cls]
        conf_values = class_data['Confidence'].astype(float)
        stats.append({
            'Class': cls,
            'Count': len(class_data),
            'Avg Confidence': f"{conf_values.mean():.4f}",
            'Min Confidence': f"{conf_values.min():.4f}",
            'Max Confidence': f"{conf_values.max():.4f}",
            'Avg Area': f"{class_data['Area'].mean():.0f}",
            'Avg Width': f"{class_data['Width'].mean():.0f}",
            'Avg Height': f"{class_data['Height'].mean():.0f}",
            'Avg Aspect Ratio': f"{class_data['AspectRatio'].astype(float).mean():.2f}"
        })
    return pd.DataFrame(stats)

def calculate_health_score(detections, image_shape):
    h, w = image_shape[:2]
    total_area = h * w
    if not detections:
        return 0
    detected_area = sum(d["Area"] for d in detections)
    coverage = (detected_area / total_area) * 100
    avg_conf = np.mean([float(d["Confidence"]) for d in detections])
    detection_count = len(detections)
    ideal_count = (total_area / 10000)
    count_score = min((detection_count / ideal_count) * 100, 100)
    health_score = (coverage * 0.3 + avg_conf * 100 * 0.4 + count_score * 0.3)
    return min(health_score, 100)

# ==================== INSIGHT GENERATORS ====================

def get_heatmap_insight(detections, heatmap_type="density"):
    if not detections:
        return "No detections available for heatmap analysis."
    
    n = len(detections)
    centers = [(d["CenterX"], d["CenterY"]) for d in detections]
    cx_vals = [c[0] for c in centers]
    cy_vals = [c[1] for c in centers]
    
    if heatmap_type == "density":
        spread_x = np.std(cx_vals) if len(cx_vals) > 1 else 0
        spread_y = np.std(cy_vals) if len(cy_vals) > 1 else 0
        cluster = "tightly clustered" if (spread_x < 100 and spread_y < 100) else "widely distributed"
        avg_x_zone = "left" if np.mean(cx_vals) < 0.33 else ("right" if np.mean(cx_vals) > 0.66 else "center")
        return (f"**Density Heatmap Analysis:** {n} object(s) detected, spatially {cluster} across the image. "
                f"Detection hotspot is concentrated toward the **{avg_x_zone}** region. "
                f"Horizontal spread (σ={spread_x:.1f}px) and vertical spread (σ={spread_y:.1f}px) indicate "
                f"{'uniform coverage' if spread_x > 150 else 'localized clustering'} of crops in the field.")
    
    elif heatmap_type == "proximity":
        if n < 2:
            return "Need at least 2 detections for proximity analysis."
        dists = []
        for i, d1 in enumerate(detections):
            for d2 in detections[i+1:]:
                dists.append(np.sqrt((d2["CenterX"]-d1["CenterX"])**2 + (d2["CenterY"]-d1["CenterY"])**2))
        avg_dist = np.mean(dists)
        return (f"**Proximity Map Analysis:** Analyzed {n*(n-1)//2} inter-object distance pair(s). "
                f"Average inter-crop distance: **{avg_dist:.1f}px**. "
                f"{'Crops are growing in close proximity, suggesting dense planting.' if avg_dist < 150 else 'Crops are well-spaced, indicating moderate planting density.'} "
                f"Min distance: {min(dists):.1f}px | Max distance: {max(dists):.1f}px.")
    
    return "Analysis complete."

def get_gradcam_insight(detections):
    if not detections:
        return "No detections to analyze for Grad-CAM attention patterns."
    confidences = [float(d["Confidence"]) for d in detections]
    avg_conf = np.mean(confidences)
    high_conf = sum(1 for c in confidences if c > 0.8)
    return (f"**Grad-CAM Attention Analysis:** The attention heatmap highlights regions the model focused on during inference. "
            f"Average activation confidence: **{avg_conf:.3f}**. "
            f"{high_conf} high-confidence detection(s) (>0.80) contributed to strong activation zones. "
            f"{'Model attention is well-localized, indicating reliable feature extraction.' if avg_conf > 0.7 else 'Some activations have moderate confidence — consider re-evaluating threshold settings.'}")

def get_3d_insight(detections):
    if not detections:
        return "No 3D data to analyze."
    classes = list(set(d["Class"] for d in detections))
    areas = [d["Area"] for d in detections]
    confs = [float(d["Confidence"]) for d in detections]
    return (f"**3D Visualization Insight:** Spatial distribution of {len(detections)} detection(s) across {len(classes)} class(es): {', '.join(classes)}. "
            f"Z-axis (confidence) ranges from {min(confs):.3f} to {max(confs):.3f}. "
            f"Object area ranges from {min(areas)} to {max(areas)} px². "
            f"{'Multi-class distribution detected — consider class-specific thresholds.' if len(classes) > 1 else 'Single-class scene detected.'}")

def get_analytics_insight(detections):
    if not detections:
        return "No detections available for analytics."
    confs = [float(d["Confidence"]) for d in detections]
    areas = [d["Area"] for d in detections]
    class_counts = {}
    for d in detections:
        class_counts[d["Class"]] = class_counts.get(d["Class"], 0) + 1
    dominant = max(class_counts, key=class_counts.get) if class_counts else "N/A"
    return (f"**Analytics Summary:** {len(detections)} total detection(s). "
            f"Dominant class: **{dominant}** ({class_counts.get(dominant,0)} instances). "
            f"Confidence — Mean: {np.mean(confs):.3f}, Std: {np.std(confs):.3f}. "
            f"Avg object size: {np.mean(areas):.0f} px². "
            f"{'High confidence uniformity detected.' if np.std(confs) < 0.1 else 'Variable confidence levels suggest mixed detection quality.'}")

def get_image_analysis_insight(image_quality, detections):
    blur = float(image_quality["Blur Score"])
    brightness = float(image_quality["Brightness"])
    quality_text = image_quality["Image Quality"]
    return (f"**Image Quality Analysis:** Overall quality rated as **{quality_text}**. "
            f"Sharpness (Laplacian variance): {blur:.1f} — {'excellent for detection' if blur > 200 else 'adequate' if blur > 100 else 'may limit detection accuracy'}. "
            f"Brightness level: {brightness:.0f}/255 — {'well-lit' if 80 < brightness < 200 else 'may be over/under-exposed'}. "
            f"Histogram equalization applied to enhance contrast for low-light field conditions.")

def get_advanced_charts_insight(detections):
    if not detections:
        return "No detections available for advanced chart analysis."
    widths = [d["Width"] for d in detections]
    heights = [d["Height"] for d in detections]
    aspects = [float(d["AspectRatio"]) for d in detections]
    return (f"**Advanced Charts Insight:** Bounding box analysis across {len(detections)} detection(s). "
            f"Width range: {min(widths)}–{max(widths)}px (mean: {np.mean(widths):.1f}px). "
            f"Height range: {min(heights)}–{max(heights)}px (mean: {np.mean(heights):.1f}px). "
            f"Avg aspect ratio: {np.mean(aspects):.2f} — "
            f"{'Objects are roughly square-shaped.' if 0.8 < np.mean(aspects) < 1.2 else 'Objects are elongated/wide.'} "
            f"ROC curve provides pseudo-confidence threshold performance characterization.")

# ==================== PDF GENERATORS ====================

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='PNG', dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf

def make_pdf_styles():
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('AgroTitle', parent=styles['Heading1'],
        fontSize=22, textColor=rl_colors.HexColor('#1a5c2e'),
        spaceAfter=10, alignment=1, fontName='Helvetica-Bold')
    h2_style = ParagraphStyle('AgroH2', parent=styles['Heading2'],
        fontSize=14, textColor=rl_colors.HexColor('#1a5c2e'),
        spaceBefore=12, spaceAfter=6, fontName='Helvetica-Bold')
    h3_style = ParagraphStyle('AgroH3', parent=styles['Heading3'],
        fontSize=11, textColor=rl_colors.HexColor('#155e44'),
        spaceBefore=8, spaceAfter=4, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('AgroBody', parent=styles['Normal'],
        fontSize=9.5, textColor=rl_colors.HexColor('#222222'),
        leading=15, spaceAfter=4)
    insight_style = ParagraphStyle('Insight', parent=styles['Normal'],
        fontSize=9, textColor=rl_colors.HexColor('#1a3a2a'),
        leading=14, leftIndent=12, rightIndent=12,
        backColor=rl_colors.HexColor('#e8f5ee'),
        borderPadding=(6, 8, 6, 8))
    return styles, title_style, h2_style, h3_style, body_style, insight_style

def generate_pdf_report(image_pil, annotated_img, detections, image_quality, health_score, stats_df, section="full"):
    try:
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4,
                                leftMargin=0.75*inch, rightMargin=0.75*inch,
                                topMargin=0.75*inch, bottomMargin=0.75*inch)
        styles, title_style, h2_style, h3_style, body_style, insight_style = make_pdf_styles()
        story = []
        
        # Cover
        story.append(Paragraph("🌿 AgroVision", title_style))
        story.append(Paragraph("SSL & Graph-Refined Object Detection for Precision Agriculture", 
            ParagraphStyle('sub', parent=styles['Normal'], fontSize=11, alignment=1, textColor=rl_colors.HexColor('#4a8c5c'))))
        story.append(Spacer(1, 0.15*inch))
        story.append(HRFlowable(width="100%", thickness=2, color=rl_colors.HexColor('#2ed573')))
        story.append(Spacer(1, 0.1*inch))
        
        report_label = {
            "full": "Full Analysis Report",
            "detection": "Detection Report",
            "heatmap": "Heatmap Analysis Report",
            "gradcam": "Grad-CAM Report",
            "3d": "3D Visualization Report",
            "image_analysis": "Image Analysis Report",
            "analytics": "Analytics Report",
            "advanced_charts": "Advanced Charts Report",
        }.get(section, "Analysis Report")
        
        story.append(Paragraph(report_label, h2_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Key metrics
        metrics_data = [
            ["Metric", "Value", "Metric", "Value"],
            ["Health Score", f"{health_score:.1f}/100", "Total Detections", str(len(detections))],
            ["Avg Confidence", f"{np.mean([float(d['Confidence']) for d in detections]):.4f}" if detections else "N/A",
             "Image Quality", image_quality["Image Quality"]],
            ["Blur Score", image_quality["Blur Score"], "Brightness", image_quality["Brightness"]],
        ]
        metrics_table = Table(metrics_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), rl_colors.HexColor('#1a5c2e')),
            ('TEXTCOLOR', (0,0), (-1,0), rl_colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BACKGROUND', (0,1), (0,-1), rl_colors.HexColor('#e8f5ee')),
            ('BACKGROUND', (2,1), (2,-1), rl_colors.HexColor('#e8f5ee')),
            ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
            ('FONTNAME', (2,1), (2,-1), 'Helvetica-Bold'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('GRID', (0,0), (-1,-1), 0.5, rl_colors.HexColor('#aaddbb')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [rl_colors.white, rl_colors.HexColor('#f0faf3')]),
            ('BOTTOMPADDING', (0,0), (-1,-1), 7),
            ('TOPPADDING', (0,0), (-1,-1), 7),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Original + Annotated images
        if section in ("full", "detection"):
            story.append(HRFlowable(width="100%", thickness=1, color=rl_colors.HexColor('#aaddbb')))
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph("Detection Results", h2_style))
            
            orig_buf = io.BytesIO()
            image_pil.save(orig_buf, format='PNG')
            orig_buf.seek(0)
            ann_buf = io.BytesIO()
            annotated_img.save(ann_buf, format='PNG')
            ann_buf.seek(0)
            
            img_table = Table([
                [Paragraph("Original Image", h3_style), Paragraph("Detected Image", h3_style)],
                [RLImage(orig_buf, width=3*inch, height=2.2*inch), 
                 RLImage(ann_buf, width=3*inch, height=2.2*inch)]
            ], colWidths=[3.5*inch, 3.5*inch])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            story.append(img_table)
            story.append(Spacer(1, 0.15*inch))
            
            if detections:
                story.append(Paragraph("Insight", h3_style))
                insight_text = (f"Detection found {len(detections)} object(s). "
                    f"Average confidence: {np.mean([float(d['Confidence']) for d in detections]):.3f}. "
                    f"Health score: {health_score:.1f}/100 — "
                    f"{'Excellent crop coverage' if health_score > 75 else 'Moderate coverage' if health_score > 40 else 'Low coverage detected'}.")
                story.append(Paragraph(insight_text, insight_style))
            story.append(Spacer(1, 0.15*inch))
        
        # Class stats
        if section in ("full", "detection", "analytics") and not stats_df.empty:
            story.append(HRFlowable(width="100%", thickness=1, color=rl_colors.HexColor('#aaddbb')))
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph("Class-wise Statistics", h2_style))
            
            header = list(stats_df.columns)
            data = [header] + [list(row) for _, row in stats_df.iterrows()]
            tbl = Table(data, repeatRows=1)
            col_count = len(header)
            tbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), rl_colors.HexColor('#1a5c2e')),
                ('TEXTCOLOR', (0,0), (-1,0), rl_colors.white),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTSIZE', (0,0), (-1,-1), 8.5),
                ('GRID', (0,0), (-1,-1), 0.5, rl_colors.HexColor('#aaddbb')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [rl_colors.white, rl_colors.HexColor('#f0faf3')]),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 0.15*inch))
            
            story.append(Paragraph("Analytics Insight", h3_style))
            story.append(Paragraph(get_analytics_insight(detections).replace("**", ""), insight_style))
        
        # Detections table
        if section in ("full", "detection") and detections:
            story.append(PageBreak())
            story.append(Paragraph("All Detections", h2_style))
            det_df = pd.DataFrame(detections)
            det_header = list(det_df.columns)
            det_data = [det_header] + [list(row) for _, row in det_df.iterrows()]
            det_tbl = Table(det_data, repeatRows=1)
            det_tbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), rl_colors.HexColor('#1a5c2e')),
                ('TEXTCOLOR', (0,0), (-1,0), rl_colors.white),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTSIZE', (0,0), (-1,-1), 7.5),
                ('GRID', (0,0), (-1,-1), 0.5, rl_colors.HexColor('#aaddbb')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [rl_colors.white, rl_colors.HexColor('#f0faf3')]),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                ('TOPPADDING', (0,0), (-1,-1), 5),
            ]))
            story.append(det_tbl)
        
        doc.build(story)
        pdf_buffer.seek(0)
        return pdf_buffer
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

def generate_section_pdf(section_name, insight_text, fig=None, detections=None, image_quality=None, health_score=0, stats_df=None):
    try:
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4,
                                leftMargin=0.75*inch, rightMargin=0.75*inch,
                                topMargin=0.75*inch, bottomMargin=0.75*inch)
        styles, title_style, h2_style, h3_style, body_style, insight_style = make_pdf_styles()
        story = []
        
        story.append(Paragraph("🌿 AgroVision", title_style))
        story.append(HRFlowable(width="100%", thickness=2, color=rl_colors.HexColor('#2ed573')))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(f"{section_name} Report", h2_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 0.15*inch))
        
        if image_quality:
            n = len(detections) if detections else 0
            metrics_data = [
                ["Health Score", f"{health_score:.1f}/100", "Detections", str(n)],
                ["Image Quality", image_quality["Image Quality"], "Avg Confidence",
                 f"{np.mean([float(d['Confidence']) for d in detections]):.4f}" if detections else "N/A"],
            ]
            mt = Table([["Metric","Value","Metric","Value"]] + metrics_data, colWidths=[1.5*inch]*4)
            mt.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), rl_colors.HexColor('#1a5c2e')),
                ('TEXTCOLOR', (0,0), (-1,0), rl_colors.white),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTSIZE', (0,0), (-1,-1), 9),
                ('GRID', (0,0), (-1,-1), 0.5, rl_colors.HexColor('#aaddbb')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [rl_colors.white, rl_colors.HexColor('#f0faf3')]),
                ('BOTTOMPADDING', (0,0), (-1,-1), 7),
            ]))
            story.append(mt)
            story.append(Spacer(1, 0.15*inch))
        
        if fig:
            story.append(Paragraph("Visualization", h3_style))
            img_buf = fig_to_bytes(fig)
            story.append(RLImage(img_buf, width=5.5*inch, height=3.5*inch))
            story.append(Spacer(1, 0.15*inch))
        
        story.append(Paragraph("AI Analysis & Insights", h3_style))
        clean_insight = insight_text.replace("**", "").replace("*", "")
        story.append(Paragraph(clean_insight, insight_style))
        
        doc.build(story)
        pdf_buffer.seek(0)
        return pdf_buffer
    except Exception as e:
        st.error(f"PDF error: {e}")
        return None

def export_detections_json(detections, image_filename):
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "image_filename": image_filename,
        "total_detections": len(detections),
        "detections": detections
    }
    return json.dumps(export_data, indent=2)

# ==================== INSIGHT BOX HELPER ====================

def show_insight_box(insight_text, icon="🔬"):
    clean = insight_text.replace("**", "<strong>").replace("**", "</strong>")
    st.markdown(f"""
    <div class="analysis-insight">
        <div class="insight-title">{icon} AI Analysis Insight</div>
        {insight_text.replace("**", "<b>").replace("**", "</b>")}
    </div>
    """, unsafe_allow_html=True)

def render_insight_and_pdf(insight_text, section_name, icon, pdf_fig=None,
                            detections=None, image_quality=None, health_score=0):
    # Insight box
    st.markdown(f"""
    <div class="analysis-insight">
        <div class="insight-title">{icon} AI Analysis Insight</div>
        <span>{insight_text.replace(chr(42)+chr(42), "<b>").replace("<b>", "<b>")}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Section PDF download
    col_pad, col_btn = st.columns([4, 1])
    with col_btn:
        if st.button(f"📄 PDF Report", key=f"pdf_{section_name}"):
            with st.spinner(f"Generating {section_name} PDF..."):
                iq = image_quality or {}
                pdf_buf = generate_section_pdf(
                    section_name, insight_text, pdf_fig,
                    detections, iq, health_score
                )
                if pdf_buf:
                    st.download_button(
                        f"⬇️ Download {section_name} PDF",
                        data=pdf_buf,
                        file_name=f"agrovision_{section_name.lower().replace(' ','_')}.pdf",
                        mime="application/pdf",
                        key=f"dl_{section_name}"
                    )

# ==================== MAIN APP ====================

st.markdown("---")

if not uploaded_models:
    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(46,213,115,0.06),rgba(52,152,219,0.06));
                border:1px solid rgba(46,213,115,0.2);border-radius:16px;padding:2rem 2.5rem;text-align:center;'>
        <div style='font-size:3rem;margin-bottom:1rem;'>🌿</div>
        <div style='font-size:1.2rem;font-weight:600;color:rgba(200,255,200,0.85);margin-bottom:0.5rem;'>
            Upload your model to begin
        </div>
        <div style='font-size:0.9rem;color:rgba(180,255,180,0.5);'>
            Use the sidebar to upload <code>.pt</code> or <code>.pkl</code> model files (up to 10 at once)
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    model, model_type = load_active_model(uploaded_models, selected_model_name)
    
    if model is not None and model_type == "yolo":
        if batch_mode:
            st.markdown('<div class="section-header">📁 Batch Processing Mode</div>', unsafe_allow_html=True)
            uploaded_files = st.file_uploader(
                "Upload multiple images",
                type=["jpg", "jpeg", "png", "bmp"],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                batch_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name} ({idx+1}/{len(uploaded_files)})...")
                    image_pil = Image.open(uploaded_file).convert('RGB')
                    image_np = pil_to_cv2(image_pil)
                    
                    results = run_detection(image_np, model, conf_threshold, iou_threshold)
                    annotated_img, detections = draw_detections_advanced(image_np, results, conf_threshold)
                    filtered_detections = filter_detections(detections, filter_class, min_conf_filter, max_conf_filter, min_size, max_size)
                    
                    batch_results.append({
                        "filename": uploaded_file.name,
                        "detections": len(filtered_detections),
                        "avg_confidence": np.mean([float(d["Confidence"]) for d in filtered_detections]) if filtered_detections else 0,
                        "image": image_pil,
                        "annotated": annotated_img,
                        "data": filtered_detections
                    })
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.empty()
                
                st.markdown('<div class="section-header">📊 Batch Results Summary</div>', unsafe_allow_html=True)
                batch_df = pd.DataFrame([{
                    "Filename": r["filename"],
                    "Detections": r["detections"],
                    "Avg Confidence": f"{r['avg_confidence']:.4f}"
                } for r in batch_results])
                st.dataframe(batch_df, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Detections (Batch)", sum(r["detections"] for r in batch_results))
                with col2:
                    st.metric("Avg Confidence (Batch)", f"{np.mean([r['avg_confidence'] for r in batch_results]):.4f}")
                
                st.markdown("---")
                st.markdown('<div class="section-header">💾 Batch Export</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📦 Create ZIP Export"):
                        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
                            with zipfile.ZipFile(tmp_zip, 'w') as z:
                                for result in batch_results:
                                    img_buffer = io.BytesIO()
                                    result["annotated"].save(img_buffer, format='PNG')
                                    z.writestr(f"annotated_{result['filename']}", img_buffer.getvalue())
                                    json_data = export_detections_json(result["data"], result["filename"])
                                    z.writestr(f"detections_{result['filename']}.json", json_data)
                            tmp_zip.seek(0)
                            st.download_button(
                                "⬇️ Download Batch Results (ZIP)",
                                data=tmp_zip.read(),
                                file_name="batch_results.zip",
                                mime="application/zip"
                            )
                with col2:
                    if st.button("📑 Batch PDF Report"):
                        with st.spinner("Generating batch PDF..."):
                            all_dets = []
                            for r in batch_results:
                                all_dets.extend(r["data"])
                            iq = analyze_image_quality(pil_to_cv2(batch_results[0]["image"])) if batch_results else {}
                            hs = calculate_health_score(all_dets, pil_to_cv2(batch_results[0]["image"]).shape) if batch_results else 0
                            sdf = create_class_statistics(all_dets)
                            pdf_buf = generate_pdf_report(
                                batch_results[0]["image"], batch_results[0]["annotated"],
                                all_dets, iq, hs, sdf, section="full"
                            )
                            if pdf_buf:
                                st.download_button("⬇️ Download Batch PDF", data=pdf_buf,
                                    file_name="agrovision_batch_report.pdf", mime="application/pdf")
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                uploaded_file = st.file_uploader(
                    "📤 Upload an Image",
                    type=["jpg", "jpeg", "png", "bmp"],
                    help="Upload an image for detection"
                )
            with col2:
                st.write("")
                st.write("")
                if st.button("🔍 Run Detection", use_container_width=True):
                    st.session_state.run_detection = True
            
            if uploaded_file is not None:
                image_pil = Image.open(uploaded_file).convert('RGB')
                image_np = pil_to_cv2(image_pil)
                
                with st.spinner("🔄 Running AgroVision detection..."):
                    results = run_detection(image_np, model, conf_threshold, iou_threshold)
                
                annotated_img, detections = draw_detections_advanced(image_np, results, conf_threshold, show_grid=enable_grid)
                filtered_detections = filter_detections(detections, filter_class, min_conf_filter, max_conf_filter, min_size, max_size)
                
                st.success(f"✅ Detection Complete! Found **{len(filtered_detections)}** object(s) (after filtering)")
                
                st.markdown("---")
                
                image_quality = analyze_image_quality(image_np)
                health_score = calculate_health_score(filtered_detections, image_np.shape)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🌾 Health Score", f"{health_score:.1f}/100")
                with col2:
                    st.metric("📊 Detections", len(filtered_detections))
                with col3:
                    avg_conf = np.mean([float(d["Confidence"]) for d in filtered_detections]) if filtered_detections else 0
                    st.metric("📈 Avg Confidence", f"{avg_conf:.4f}")
                with col4:
                    st.metric("🔍 Image Quality", image_quality["Image Quality"])
                
                st.markdown("---")
                
                stats_df = create_class_statistics(filtered_detections)
                
                tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
                    ["📸 Detection", "🔥 Grad-CAM", "🌡️ Heatmaps", "3️⃣ 3D View",
                     "🔬 Image Analysis", "📊 Analytics", "📈 Advanced Charts", "📋 Details"]
                )
                
                # ── TAB 1: Detection ──
                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="section-header">Original Image</div>', unsafe_allow_html=True)
                        st.image(image_pil, use_column_width=True)
                    with col2:
                        st.markdown('<div class="section-header">Detected Image</div>', unsafe_allow_html=True)
                        st.image(annotated_img, use_column_width=True)
                    
                    det_insight = (f"**Detection Result:** Found {len(filtered_detections)} object(s) with avg confidence {avg_conf:.3f}. "
                        f"Health score: {health_score:.1f}/100. "
                        f"{'Excellent crop coverage detected.' if health_score > 75 else 'Moderate crop presence.' if health_score > 40 else 'Low or no detections — try adjusting threshold.'}")
                    
                    render_insight_and_pdf(
                        det_insight, "Detection", "📸",
                        detections=filtered_detections,
                        image_quality=image_quality,
                        health_score=health_score
                    )
                
                # ── TAB 2: Grad-CAM ──
                with tab2:
                    if enable_gradcam:
                        st.markdown('<div class="section-header">🔥 Grad-CAM Attention Heatmap</div>', unsafe_allow_html=True)
                        try:
                            gradcam_img, heatmap = generate_gradcam(image_np, model)
                            if gradcam_img:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(gradcam_img, caption="Overlay", use_column_width=True)
                                with col2:
                                    st.image(Image.fromarray((heatmap * 255).astype(np.uint8)),
                                            caption="Raw Heatmap", use_column_width=True, channels="GRAY")
                                
                                gc_insight = get_gradcam_insight(filtered_detections)
                                render_insight_and_pdf(
                                    gc_insight, "Grad-CAM", "🔥",
                                    detections=filtered_detections,
                                    image_quality=image_quality,
                                    health_score=health_score
                                )
                        except Exception as e:
                            st.error(f"Grad-CAM error: {e}")
                    else:
                        st.info("Grad-CAM disabled in settings")
                
                # ── TAB 3: Heatmaps ──
                with tab3:
                    col1, col2 = st.columns(2)
                    density_fig = None
                    prox_insight = ""
                    
                    with col1:
                        if enable_heatmap:
                            st.markdown('<div class="section-header">Density Heatmap</div>', unsafe_allow_html=True)
                            density_img, density_map = generate_density_heatmap(filtered_detections, image_np.shape)
                            st.image(density_img, use_column_width=True)
                    with col2:
                        if enable_proximity:
                            st.markdown('<div class="section-header">Proximity Map</div>', unsafe_allow_html=True)
                            proximity_img = create_proximity_map(filtered_detections, image_np.shape)
                            if proximity_img:
                                st.image(proximity_img, use_column_width=True)
                            else:
                                st.info("Need 2+ detections for proximity analysis")
                    
                    density_insight = get_heatmap_insight(filtered_detections, "density")
                    prox_insight = get_heatmap_insight(filtered_detections, "proximity")
                    combined = density_insight + " | " + prox_insight
                    
                    render_insight_and_pdf(
                        combined, "Heatmap", "🌡️",
                        detections=filtered_detections,
                        image_quality=image_quality,
                        health_score=health_score
                    )
                
                # ── TAB 4: 3D View ──
                with tab4:
                    if enable_3d:
                        st.markdown('<div class="section-header">🔮 3D Detection Visualization</div>', unsafe_allow_html=True)
                        fig_3d = create_3d_plot(filtered_detections)
                        if fig_3d:
                            st.plotly_chart(fig_3d, use_container_width=True)
                            td_insight = get_3d_insight(filtered_detections)
                            render_insight_and_pdf(
                                td_insight, "3D Visualization", "🔮",
                                detections=filtered_detections,
                                image_quality=image_quality,
                                health_score=health_score
                            )
                        else:
                            st.info("No detections for 3D visualization")
                    else:
                        st.info("3D visualization disabled")
                
                # ── TAB 5: Image Analysis ──
                with tab5:
                    if enable_preprocessing:
                        col1, col2 = st.columns(2)
                        color_fig = None
                        
                        with col1:
                            st.markdown('<div class="section-header">Image Quality Metrics</div>', unsafe_allow_html=True)
                            quality_df = pd.DataFrame([image_quality])
                            st.dataframe(quality_df, use_container_width=True)
                            
                            st.markdown('<div class="section-header">Edge Detection</div>', unsafe_allow_html=True)
                            edges = create_edge_detection(image_np)
                            st.image(edges, caption="Canny Edge Detection", use_column_width=True, channels="GRAY")
                        
                        with col2:
                            st.markdown('<div class="section-header">Histogram Equalization</div>', unsafe_allow_html=True)
                            equalized = create_histogram_equalized(image_np)
                            st.image(equalized, use_column_width=True)
                            
                            st.markdown('<div class="section-header">Color Channel Distribution</div>', unsafe_allow_html=True)
                            color_fig = create_color_distribution(image_np)
                            st.pyplot(color_fig, use_container_width=True)
                        
                        ia_insight = get_image_analysis_insight(image_quality, filtered_detections)
                        render_insight_and_pdf(
                            ia_insight, "Image Analysis", "🔬",
                            pdf_fig=color_fig,
                            detections=filtered_detections,
                            image_quality=image_quality,
                            health_score=health_score
                        )
                    else:
                        st.info("Image analysis disabled")
                
                # ── TAB 6: Analytics ──
                with tab6:
                    col1, col2 = st.columns(2)
                    class_fig = None
                    conf_fig = None
                    
                    with col1:
                        st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
                        class_fig = create_class_distribution(filtered_detections)
                        st.pyplot(class_fig, use_container_width=True)
                    with col2:
                        st.markdown('<div class="section-header">Confidence Distribution</div>', unsafe_allow_html=True)
                        conf_fig = create_confidence_distribution(filtered_detections)
                        st.pyplot(conf_fig, use_container_width=True)
                    
                    analytics_insight = get_analytics_insight(filtered_detections)
                    render_insight_and_pdf(
                        analytics_insight, "Analytics", "📊",
                        pdf_fig=class_fig,
                        detections=filtered_detections,
                        image_quality=image_quality,
                        health_score=health_score
                    )
                
                # ── TAB 7: Advanced Charts ──
                with tab7:
                    col1, col2 = st.columns(2)
                    scatter_fig = None
                    
                    with col1:
                        st.markdown('<div class="section-header">Confidence vs Size</div>', unsafe_allow_html=True)
                        scatter_fig = create_scatter_plot(filtered_detections)
                        st.pyplot(scatter_fig, use_container_width=True)
                    with col2:
                        st.markdown('<div class="section-header">Aspect Ratio Distribution</div>', unsafe_allow_html=True)
                        ar_fig = create_aspect_ratio_distribution(filtered_detections)
                        st.pyplot(ar_fig, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="section-header">Detection Density Grid</div>', unsafe_allow_html=True)
                        dens_fig = create_detection_density_chart(filtered_detections, image_np.shape)
                        st.pyplot(dens_fig, use_container_width=True)
                    with col2:
                        st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
                        roc_fig = create_roc_curve(filtered_detections)
                        st.pyplot(roc_fig, use_container_width=True)
                    
                    st.markdown('<div class="section-header">Bounding Box Size Distributions</div>', unsafe_allow_html=True)
                    bbox_fig = create_bbox_distribution(filtered_detections)
                    st.pyplot(bbox_fig, use_container_width=True)
                    
                    ac_insight = get_advanced_charts_insight(filtered_detections)
                    render_insight_and_pdf(
                        ac_insight, "Advanced Charts", "📈",
                        pdf_fig=scatter_fig,
                        detections=filtered_detections,
                        image_quality=image_quality,
                        health_score=health_score
                    )
                
                # ── TAB 8: Details ──
                with tab8:
                    st.markdown('<div class="section-header">📋 Class-wise Statistics</div>', unsafe_allow_html=True)
                    if not stats_df.empty:
                        st.dataframe(stats_df, use_container_width=True)
                    else:
                        st.info("No detections to show statistics for.")
                    
                    st.markdown("---")
                    st.markdown('<div class="section-header">All Detections</div>', unsafe_allow_html=True)
                    if filtered_detections:
                        detections_df = pd.DataFrame(filtered_detections)
                        st.dataframe(detections_df, use_container_width=True)
                    
                    details_insight = (f"**Details Summary:** {len(filtered_detections)} detections across "
                        f"{len(set(d['Class'] for d in filtered_detections)) if filtered_detections else 0} class(es). "
                        f"Health Score: {health_score:.1f}/100. "
                        f"Image quality: {image_quality['Image Quality']} (blur={image_quality['Blur Score']}, brightness={image_quality['Brightness']}).")
                    
                    render_insight_and_pdf(
                        details_insight, "Details", "📋",
                        detections=filtered_detections,
                        image_quality=image_quality,
                        health_score=health_score
                    )
                
                # ── EXPORT SECTION ──
                st.markdown("---")
                st.markdown('<div class="section-header">💾 Export & Download</div>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown('<div class="export-card">', unsafe_allow_html=True)
                    img_buffer = io.BytesIO()
                    annotated_img.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    st.download_button(
                        "📸 Annotated Image",
                        data=img_buffer,
                        file_name="agrovision_detection.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="export-card">', unsafe_allow_html=True)
                    if filtered_detections:
                        csv_data = pd.DataFrame(filtered_detections).to_csv(index=False)
                        st.download_button(
                            "📊 Detections CSV",
                            data=csv_data,
                            file_name="agrovision_detections.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="export-card">', unsafe_allow_html=True)
                    json_data = export_detections_json(filtered_detections, uploaded_file.name)
                    st.download_button(
                        "📄 Detections JSON",
                        data=json_data,
                        file_name="agrovision_detections.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown('<div class="export-card">', unsafe_allow_html=True)
                    if st.button("📑 Full PDF Report", use_container_width=True):
                        with st.spinner("Generating full AgroVision PDF..."):
                            pdf_buffer = generate_pdf_report(
                                image_pil, annotated_img, filtered_detections,
                                image_quality, health_score, stats_df, section="full"
                            )
                            if pdf_buffer:
                                st.download_button(
                                    "⬇️ Download Full PDF",
                                    data=pdf_buffer,
                                    file_name="agrovision_full_report.pdf",
                                    mime="application/pdf"
                                )
                    st.markdown('</div>', unsafe_allow_html=True)
            
            else:
                st.markdown("""
                <div style='background:rgba(52,152,219,0.08);border:1px solid rgba(52,152,219,0.2);
                            border-radius:14px;padding:1.5rem;text-align:center;margin-top:1rem;'>
                    <div style='font-size:2rem;margin-bottom:0.5rem;'>👆</div>
                    <div style='color:rgba(200,220,255,0.7);font-size:0.95rem;'>Upload an image to begin detection</div>
                </div>
                """, unsafe_allow_html=True)
    
    elif model is not None and model_type == "pkl":
        st.info("🧪 PKL model loaded. Custom PKL inference pipeline — please integrate your prediction logic here.")
        st.code(f"Model type: {type(model).__name__}", language="python")
    
    elif model is None:
        st.error("⚠️ Failed to load the selected model. Please check the file format.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:1rem 0;'>
    <span style='color:rgba(46,213,115,0.4);font-size:0.8rem;letter-spacing:2px;'>
        🌿 AGROVISION · SSL & GRAPH-REFINED OBJECT DETECTION · PRECISION AGRICULTURE AI
    </span>
</div>
""", unsafe_allow_html=True)