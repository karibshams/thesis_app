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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp {
        background: linear-gradient(135deg, #0a0f0a 0%, #0d1f0d 40%, #0a1a1f 100%);
        min-height: 100vh;
    }
    /* ── Logo Hero ── */
    .agrovision-hero {
        background: linear-gradient(135deg, #0d2b1a 0%, #0a1f2e 50%, #1a0d2b 100%);
        border: 1px solid rgba(46,213,115,0.2);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5), 0 0 80px rgba(46,213,115,0.05);
    }
    .agrovision-hero::before {
        content: '';
        position: absolute; top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(46,213,115,0.06) 0%, transparent 50%),
                    radial-gradient(circle at 70% 50%, rgba(52,152,219,0.06) 0%, transparent 50%);
        pointer-events: none;
    }
    .agrovision-logo-wrap {
        display: flex; align-items: center; gap: 18px; margin-bottom: 0.3rem;
    }
    .agrovision-logo-svg {
        flex-shrink: 0;
        filter: drop-shadow(0 0 12px rgba(46,213,115,0.4));
    }
    .agrovision-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #2ed573, #3498db, #a29bfe);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin: 0; line-height: 1.1; letter-spacing: -1px;
    }
    .agrovision-subtitle {
        font-size: 0.95rem; color: rgba(200,255,200,0.55);
        margin-top: 0.3rem; font-weight: 300; letter-spacing: 0.5px;
    }
    .agrovision-badge {
        display: inline-block;
        background: rgba(46,213,115,0.15);
        border: 1px solid rgba(46,213,115,0.4);
        color: #2ed573; padding: 4px 14px; border-radius: 20px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 1px;
        text-transform: uppercase; margin-bottom: 0.6rem;
    }
    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1f0d 0%, #0a1420 100%);
        border-right: 1px solid rgba(46,213,115,0.15);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #2ed573 !important; font-size: 0.9rem !important;
        text-transform: uppercase; letter-spacing: 1px; font-weight: 600;
    }
    /* ── Metric Cards ── */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(13,43,26,0.8), rgba(10,26,47,0.8));
        border: 1px solid rgba(46,213,115,0.2); border-radius: 16px;
        padding: 1.2rem 1.5rem; transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    [data-testid="metric-container"]:hover {
        border-color: rgba(46,213,115,0.5);
        box-shadow: 0 8px 30px rgba(46,213,115,0.1);
        transform: translateY(-2px);
    }
    [data-testid="metric-container"] label {
        color: rgba(180,255,180,0.7) !important;
        font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 1px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #2ed573 !important; font-weight: 700 !important; font-size: 1.8rem !important;
    }
    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #1a7a3c, #155e75);
        color: white; border: 1px solid rgba(46,213,115,0.4); border-radius: 10px;
        font-weight: 600; letter-spacing: 0.5px; transition: all 0.3s ease; padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #22a84d, #1a7a9a);
        border-color: #2ed573; box-shadow: 0 0 20px rgba(46,213,115,0.3); transform: translateY(-1px);
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #0d4429, #0a2d3d) !important;
        color: #2ed573 !important; border: 1px solid rgba(46,213,115,0.5) !important;
        border-radius: 10px !important; font-weight: 600 !important; transition: all 0.3s ease;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #1a7a3c, #155e75) !important;
        box-shadow: 0 0 20px rgba(46,213,115,0.2) !important;
    }
    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(13,31,13,0.8); border-radius: 14px;
        padding: 5px; gap: 4px; border: 1px solid rgba(46,213,115,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; border-radius: 10px;
        color: rgba(180,255,180,0.6); font-weight: 500;
        font-size: 0.85rem; transition: all 0.2s; padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(46,213,115,0.2), rgba(52,152,219,0.2)) !important;
        color: #2ed573 !important; border: 1px solid rgba(46,213,115,0.3) !important;
    }
    /* ── Alerts ── */
    .stAlert { border-radius: 12px; }
    .stSuccess { background: rgba(46,213,115,0.1) !important; border-color: rgba(46,213,115,0.4) !important; border-radius: 12px !important; color: #2ed573 !important; }
    .stWarning { border-radius: 12px !important; }
    .stInfo { border-radius: 12px !important; }
    /* ── DataFrames ── */
    .dataframe { border-radius: 12px !important; overflow: hidden; }
    /* ── Section headers ── */
    .section-header {
        background: linear-gradient(135deg, rgba(46,213,115,0.08), rgba(52,152,219,0.08));
        border-left: 4px solid #2ed573; border-radius: 0 12px 12px 0;
        padding: 0.8rem 1.2rem; margin: 1.5rem 0 1rem 0;
        color: rgba(200,255,200,0.9); font-weight: 600; font-size: 1.05rem;
    }
    /* ── Insight box ── */
    .analysis-insight {
        background: linear-gradient(135deg, rgba(13,43,26,0.6), rgba(10,26,47,0.6));
        border: 1px solid rgba(46,213,115,0.15); border-radius: 14px;
        padding: 1.2rem 1.5rem; margin-top: 1rem;
        font-size: 0.9rem; color: rgba(200,255,200,0.8); line-height: 1.7;
    }
    .analysis-insight strong, .analysis-insight b { color: #2ed573; }
    .insight-title {
        font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px;
        color: #2ed573; font-weight: 700; margin-bottom: 0.6rem;
        display: flex; align-items: center; gap: 6px;
    }
    .plain-explanation {
        background: rgba(46,213,115,0.05);
        border: 1px solid rgba(46,213,115,0.12);
        border-radius: 10px; padding: 0.8rem 1.1rem;
        margin-top: 0.6rem; font-size: 0.85rem;
        color: rgba(180,255,200,0.65); font-style: italic;
    }
    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(46,213,115,0.3) !important;
        border-radius: 14px !important; background: rgba(13,43,26,0.3) !important; transition: all 0.3s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(46,213,115,0.6) !important; background: rgba(13,43,26,0.5) !important;
    }
    /* ── Progress bar ── */
    .stProgress > div > div { background: linear-gradient(90deg, #2ed573, #3498db) !important; border-radius: 10px; }
    /* ── Export card ── */
    .export-card {
        background: rgba(13,43,26,0.4); border: 1px solid rgba(46,213,115,0.15);
        border-radius: 16px; padding: 1.5rem; text-align: center; transition: all 0.3s;
    }
    .export-card:hover { border-color: rgba(46,213,115,0.4); background: rgba(13,43,26,0.6); }
    /* ── Image result card ── */
    .image-card {
        background: linear-gradient(135deg, rgba(13,43,26,0.5), rgba(10,26,47,0.5));
        border: 1px solid rgba(46,213,115,0.18); border-radius: 16px;
        padding: 1.2rem; margin-bottom: 1rem;
    }
    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0a0f0a; }
    ::-webkit-scrollbar-thumb { background: rgba(46,213,115,0.3); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(46,213,115,0.6); }
    hr { border-color: rgba(46,213,115,0.1) !important; }
</style>
""", unsafe_allow_html=True)

# ==================== LOGO HERO ====================
st.markdown("""
<div class="agrovision-hero">
    <div class="agrovision-badge">🌿 Precision Agriculture AI</div>
    <div class="agrovision-logo-wrap">
        <svg class="agrovision-logo-svg" width="62" height="62" viewBox="0 0 62 62" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="31" cy="31" r="30" stroke="rgba(46,213,115,0.35)" stroke-width="1.5"/>
            <circle cx="31" cy="31" r="24" stroke="rgba(46,213,115,0.15)" stroke-width="1"/>
            <!-- Leaf body -->
            <path d="M31 50 C31 50 14 40 14 26 C14 17 21 10 31 10 C41 10 48 17 48 26 C48 40 31 50 31 50Z"
                  fill="rgba(46,213,115,0.12)" stroke="#2ed573" stroke-width="1.8" stroke-linejoin="round"/>
            <!-- Stem -->
            <line x1="31" y1="50" x2="31" y2="23" stroke="#2ed573" stroke-width="1.8" stroke-linecap="round"/>
            <!-- Left vein -->
            <path d="M31 32 C27 29 22 27 19 22" stroke="rgba(46,213,115,0.6)" stroke-width="1.2" stroke-linecap="round"/>
            <!-- Right vein -->
            <path d="M31 36 C35 33 40 31 43 26" stroke="rgba(46,213,115,0.6)" stroke-width="1.2" stroke-linecap="round"/>
            <!-- Detection box overlay hint -->
            <rect x="23" y="18" width="10" height="10" rx="2" stroke="rgba(52,152,219,0.7)" stroke-width="1.2" fill="none" stroke-dasharray="2 1"/>
            <rect x="35" y="26" width="8" height="8" rx="2" stroke="rgba(52,152,219,0.5)" stroke-width="1" fill="none" stroke-dasharray="2 1"/>
        </svg>
        <div>
            <h1 class="agrovision-title">AgroVision</h1>
            <p style="font-size:0.85rem;font-weight:500;color:rgba(200,255,200,0.7);margin:0.15rem 0 0;">
                SSL &amp; Graph-Refined Object Detection · Sunflower &amp; Rice 
            </p>
        </div>
    </div>
    <p class="agrovision-subtitle">
        Upload one or more crop images · Get instant AI-powered detection, health scores &amp; detailed reports
    </p>
</div>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'roi_mode' not in st.session_state:
    st.session_state.roi_mode = False
if 'roi_points' not in st.session_state:
    st.session_state.roi_points = []
if 'loaded_models' not in st.session_state:
    st.session_state.loaded_models = {}

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    st.markdown("**📤 Upload Models** (up to 10)")
    st.caption("`.pt` (YOLOv) · `.pkl` (scikit-learn)")
    uploaded_models = st.file_uploader(
        "Upload model files", type=["pt", "pkl"],
        accept_multiple_files=True, label_visibility="collapsed"
    )
    if uploaded_models and len(uploaded_models) > 10:
        st.warning("⚠️ Max 10 models. Only first 10 used.")
        uploaded_models = uploaded_models[:10]

    selected_model_name = None
    if uploaded_models:
        model_names = [f.name for f in uploaded_models]
        selected_model_name = st.selectbox("🎯 Active Model", model_names)
        st.caption(f"✅ {len(uploaded_models)} model(s) loaded")

    st.markdown("---")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    iou_threshold  = st.slider("IOU Threshold",         0.1, 1.0, 0.45, 0.05)

    st.markdown("---")
    st.markdown("### 📊 Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        enable_gradcam      = st.checkbox("Grad-CAM",       value=True)
        enable_heatmap      = st.checkbox("Density Map",    value=True)
        enable_3d           = st.checkbox("3D View",        value=True)
    with col2:
        enable_preprocessing = st.checkbox("Image Analysis", value=True)
        enable_proximity     = st.checkbox("Proximity Map",  value=True)
        enable_grid          = st.checkbox("Grid Overlay",   value=True)

    st.markdown("---")
    st.markdown("### 🎯 Detection Filters")
    filter_class    = st.multiselect("Filter by Class", ["All","Sunflower","Rice"], default=["All"])
    min_conf_filter = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)
    max_conf_filter = st.slider("Max Confidence", 0.0, 1.0, 1.0, 0.05)
    min_size        = st.number_input("Min Area (px²)", value=0, min_value=0)
    max_size        = st.number_input("Max Area (px²)", value=10_000_000, min_value=0)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;padding:0.5rem;'>
        <div style='font-size:0.7rem;color:rgba(46,213,115,0.5);letter-spacing:1px;'>AGROVISION v2.1</div>
        <div style='font-size:0.65rem;color:rgba(180,255,180,0.3);margin-top:3px;'>Precision Agriculture AI</div>
    </div>
    """, unsafe_allow_html=True)

# ==================== HELPERS ====================

@st.cache_resource
def load_pt_model(model_bytes, model_name):
    try:
        tmp_path = f"temp_model_{model_name.replace(' ','_')}.pt"
        with open(tmp_path, "wb") as f:
            f.write(model_bytes)
        return YOLO(tmp_path), "yolo"
    except Exception as e:
        st.error(f"Error loading .pt model: {e}")
        return None, None

@st.cache_resource
def load_pkl_model(model_bytes, model_name):
    try:
        return pickle.loads(model_bytes), "pkl"
    except Exception as e:
        st.error(f"Error loading .pkl model: {e}")
        return None, None

def load_active_model(uploaded_models, selected_model_name):
    if not uploaded_models or not selected_model_name:
        return None, None
    for f in uploaded_models:
        if f.name == selected_model_name:
            b = f.getvalue()
            if f.name.endswith(".pt"):
                return load_pt_model(b, f.name)
            elif f.name.endswith(".pkl"):
                return load_pkl_model(b, f.name)
    return None, None

def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def run_detection(image_np, model, conf, iou):
    return model(image_np, conf=conf, iou=iou, imgsz=640)

def filter_detections(detections, class_filter, min_conf, max_conf, min_area, max_area):
    filtered = []
    for det in detections:
        conf = float(det["Confidence"])
        area = det["Area"]
        cls  = det["Class"]
        if "All" not in class_filter and cls not in class_filter:
            continue
        if not (min_conf <= conf <= max_conf):
            continue
        if not (min_area <= area <= max_area):
            continue
        filtered.append(det)
    return filtered

# ==================== VISUALIZATIONS ====================

def draw_detections_advanced(image_np, results, conf_threshold, show_grid=False):
    image_annotated = image_np.copy()
    h, w = image_np.shape[:2]
    detections = []
    CLASS_COLORS = {"Sunflower": (0,200,100), "Rice": (52,152,219)}
    DEFAULT_COLOR = (46,213,115)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf     = float(box.conf[0])
            cls_name = result.names[int(box.cls[0])]
            color    = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

            cv2.rectangle(image_annotated, (x1,y1), (x2,y2), color, 2)
            acc = 12
            for p1, p2 in [((x1,y1),(x1+acc,y1)),((x1,y1),(x1,y1+acc)),
                            ((x2,y1),(x2-acc,y1)),((x2,y1),(x2,y1+acc)),
                            ((x1,y2),(x1+acc,y2)),((x1,y2),(x1,y2-acc)),
                            ((x2,y2),(x2-acc,y2)),((x2,y2),(x2,y2-acc))]:
                cv2.line(image_annotated, p1, p2, color, 4)

            label = f"{cls_name}: {conf:.2f}"
            lsz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(image_annotated, (x1,y1-28), (x1+lsz[0]+8,y1), color, -1)
            cv2.putText(image_annotated, label, (x1+4,y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)

            cx, cy = (x1+x2)//2, (y1+y2)//2
            cv2.circle(image_annotated, (cx,cy), 4, (255,255,255), -1)
            cv2.circle(image_annotated, (cx,cy), 4, color, 2)

            detections.append({
                "Class": cls_name, "Confidence": f"{conf:.4f}",
                "X1": x1, "Y1": y1, "X2": x2, "Y2": y2,
                "CenterX": cx, "CenterY": cy,
                "Width": x2-x1, "Height": y2-y1,
                "Area": (x2-x1)*(y2-y1),
                "AspectRatio": f"{(x2-x1)/max(y2-y1,1):.2f}"
            })

    if show_grid:
        gs = 64
        for i in range(0, h, gs):
            cv2.line(image_annotated, (0,i), (w,i), (46,80,46), 1)
        for i in range(0, w, gs):
            cv2.line(image_annotated, (i,0), (i,h), (46,80,46), 1)

    return cv2_to_pil(image_annotated), detections

def generate_density_heatmap(detections, image_shape):
    h, w = image_shape[:2]
    heatmap = np.zeros((h,w), dtype=np.float32)
    for det in detections:
        x1,y1,x2,y2 = det["X1"],det["Y1"],det["X2"],det["Y2"]
        heatmap[y1:y2,x1:x2] += float(det["Confidence"])
    if heatmap.max() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    heatmap_colored = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return Image.fromarray(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)), heatmap

def generate_gradcam(image_np, model):
    try:
        h, w = image_np.shape[:2]
        heatmap = np.zeros((h,w), dtype=np.float32)
        with torch.no_grad():
            results = model(image_np, verbose=False)
        for result in results:
            for box in result.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                heatmap[y1:y2,x1:x2] += float(box.conf[0])
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
        heatmap_colored = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(image_np, 0.6, heatmap_colored, 0.4, 0)
        return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)), heatmap
    except Exception:
        return None, None

def create_proximity_map(detections, image_shape):
    h, w = image_shape[:2]
    proximity_map = np.zeros((h,w), dtype=np.float32)
    if len(detections) < 2:
        return None
    for i, d1 in enumerate(detections):
        for d2 in detections[i+1:]:
            dist = np.sqrt((d2["CenterX"]-d1["CenterX"])**2 + (d2["CenterY"]-d1["CenterY"])**2)
            cv2.line(proximity_map, (d1["CenterX"],d1["CenterY"]), (d2["CenterX"],d2["CenterY"]), dist, 2)
    if proximity_map.max() > 0:
        proximity_map = (proximity_map - proximity_map.min()) / (proximity_map.max() - proximity_map.min() + 1e-6)
    pc = cv2.applyColorMap((proximity_map*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    return Image.fromarray(cv2.cvtColor(pc, cv2.COLOR_BGR2RGB))

def create_3d_plot(detections):
    if not detections:
        return None
    df = pd.DataFrame(detections)
    df["Confidence_float"] = df["Confidence"].astype(float)
    fig = px.scatter_3d(
        df, x="CenterX", y="CenterY", z="Confidence_float",
        color="Class", size="Area",
        hover_data=["Confidence","Area"],
        title="3D Detection Map",
        labels={"CenterX":"X Pos","CenterY":"Y Pos","Confidence_float":"Confidence"},
        color_discrete_map={"Sunflower":"#2ed573","Rice":"#3498db"}
    )
    fig.update_layout(
        height=550, paper_bgcolor='rgba(10,20,10,0)', plot_bgcolor='rgba(10,20,10,0)',
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
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    contrast   = np.std(hist)
    brightness = np.mean(gray)
    return {
        "Blur Score":    f"{blur_score:.2f}",
        "Contrast":      f"{contrast:.2f}",
        "Brightness":    f"{brightness:.0f}",
        "Image Quality": "Good" if blur_score > 100 else "Fair" if blur_score > 50 else "Poor"
    }

def create_edge_detection(image_np):
    gray  = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return Image.fromarray(edges)

def create_histogram_equalized(image_np):
    lab  = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l    = cv2.equalizeHist(l)
    lab  = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

# ── matplotlib RGBA as tuples (NOT CSS strings) ────────────────────
_GREEN      = (0.180, 0.835, 0.451)   # rgb(46,213,115)
_GREEN_02   = (*_GREEN, 0.20)
_GREEN_015  = (*_GREEN, 0.15)
_GREEN_03   = (*_GREEN, 0.30)
_GREEN_08   = (*_GREEN, 0.80)
_RED_08     = (0.906, 0.298, 0.235, 0.80)
_BLUE_08    = (0.204, 0.596, 0.859, 0.80)
_PURPLE_08  = (0.635, 0.608, 0.996, 0.80)

def dark_chart_style():
    plt.rcParams.update({
        'figure.facecolor': '#0a140a',
        'axes.facecolor':   '#0d1f0d',
        'axes.edgecolor':   _GREEN_02,     # FIX: was 'rgba(...)' string
        'axes.labelcolor':  '#8fe8b0',
        'xtick.color':      '#8fe8b0',
        'ytick.color':      '#8fe8b0',
        'text.color':       '#b0e8b0',
        'grid.color':       _GREEN_015,    # FIX: was 'rgba(...)' string
        'grid.alpha':       0.3,
    })

def create_color_distribution(image_np):
    dark_chart_style()
    fig, ax = plt.subplots(figsize=(10,5), facecolor='#0a140a')
    ax.set_facecolor('#0d1f0d')
    for i, (col, label) in enumerate(zip(['#3498db','#2ed573','#e74c3c'], ['Blue','Green','Red'])):
        hist = cv2.calcHist([image_np], [i], None, [256], [0,256])
        ax.fill_between(range(256), hist.ravel(), alpha=0.3, color=col)
        ax.plot(hist, color=col, label=label, linewidth=1.5)
    ax.set_xlabel("Pixel Intensity", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("Color Channel Distribution", fontsize=13, fontweight='bold', color='#2ed573', pad=12)
    ax.legend(facecolor='#0d1f0d', edgecolor=_GREEN_03)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    return fig

def create_confidence_distribution(detections):
    dark_chart_style()
    fig, ax = plt.subplots(figsize=(10,6), facecolor='#0a140a')
    ax.set_facecolor('#0d1f0d')
    confidences = [float(d["Confidence"]) for d in detections]
    if confidences:
        ax.hist(confidences, bins=15, color='#2ed573', alpha=0.6,
                edgecolor=_GREEN_08, linewidth=0.8)
        ax.axvline(np.mean(confidences),   color='#e74c3c', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(confidences):.3f}')
        ax.axvline(np.median(confidences), color='#f39c12', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(confidences):.3f}')
        ax.set_xlabel("Confidence Score", fontsize=11)
        ax.set_ylabel("Frequency",        fontsize=11)
        ax.set_title("Confidence Score Distribution", fontsize=13, fontweight='bold', color='#2ed573', pad=12)
        ax.legend(facecolor='#0d1f0d', edgecolor=_GREEN_03)
        ax.grid(True, alpha=0.15)
    plt.tight_layout()
    return fig

def create_class_distribution(detections):
    dark_chart_style()
    class_counts = {}
    for det in detections:
        class_counts[det["Class"]] = class_counts.get(det["Class"], 0) + 1
    fig, ax = plt.subplots(figsize=(7,6), facecolor='#0a140a')
    ax.set_facecolor('#0d1f0d')
    if class_counts:
        palette = ['#2ed573','#3498db','#a29bfe','#fd79a8','#fdcb6e']
        wedges, texts, autotexts = ax.pie(
            class_counts.values(), labels=class_counts.keys(),
            autopct='%1.1f%%', colors=palette[:len(class_counts)],
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.6, edgecolor='#0a140a', linewidth=3)
        )
        for t in texts:
            t.set_color('#b0e8b0'); t.set_fontsize(11)
        for a in autotexts:
            a.set_color('white'); a.set_fontweight('bold'); a.set_fontsize(10)
        ax.set_title("Class Distribution", fontsize=13, fontweight='bold', color='#2ed573', pad=12)
    plt.tight_layout()
    return fig

def create_scatter_plot(detections):
    dark_chart_style()
    fig, ax = plt.subplots(figsize=(10,6), facecolor='#0a140a')
    ax.set_facecolor('#0d1f0d')
    if detections:
        areas  = [d["Area"] for d in detections]
        confs  = [float(d["Confidence"]) for d in detections]
        classes = [d["Class"] for d in detections]
        palette = ['#2ed573','#3498db','#a29bfe','#fd79a8']
        uclasses = list(set(classes))
        cmap = {c: palette[i % len(palette)] for i, c in enumerate(uclasses)}
        ax.scatter(areas, confs, s=120, c=[cmap[c] for c in classes],
                   alpha=0.75, edgecolors='white', linewidth=0.8)
        ax.set_xlabel("Bounding Box Area (pixels)", fontsize=11)
        ax.set_ylabel("Confidence Score", fontsize=11)
        ax.set_title("Confidence vs Bounding Box Size", fontsize=13, fontweight='bold', color='#2ed573', pad=12)
        ax.grid(True, alpha=0.15)
        handles = [plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=cmap[c],markersize=8,label=c)
                   for c in uclasses]
        ax.legend(handles=handles, loc='best', facecolor='#0d1f0d', edgecolor=_GREEN_03)
    plt.tight_layout()
    return fig

def create_bbox_distribution(detections):
    dark_chart_style()
    fig, axes = plt.subplots(1, 2, figsize=(14,5), facecolor='#0a140a')
    if detections:
        widths  = [d["Width"]  for d in detections]
        heights = [d["Height"] for d in detections]
        axes[0].set_facecolor('#0d1f0d')
        axes[0].hist(widths,  bins=10, color='#e74c3c', alpha=0.65, edgecolor=_RED_08,    linewidth=0.8)
        axes[0].set_xlabel("Width (pixels)",  fontsize=10)
        axes[0].set_ylabel("Frequency",       fontsize=10)
        axes[0].set_title("Bounding Box Width Distribution",  fontsize=12, fontweight='bold', color='#2ed573', pad=10)
        axes[0].grid(True, alpha=0.15)
        axes[1].set_facecolor('#0d1f0d')
        axes[1].hist(heights, bins=10, color='#3498db', alpha=0.65, edgecolor=_BLUE_08,   linewidth=0.8)
        axes[1].set_xlabel("Height (pixels)", fontsize=10)
        axes[1].set_ylabel("Frequency",       fontsize=10)
        axes[1].set_title("Bounding Box Height Distribution", fontsize=12, fontweight='bold', color='#2ed573', pad=10)
        axes[1].grid(True, alpha=0.15)
    plt.tight_layout()
    return fig

def create_aspect_ratio_distribution(detections):
    dark_chart_style()
    fig, ax = plt.subplots(figsize=(10,6), facecolor='#0a140a')
    ax.set_facecolor('#0d1f0d')
    if detections:
        ars = [float(d["AspectRatio"]) for d in detections]
        ax.hist(ars, bins=15, color='#a29bfe', alpha=0.65, edgecolor=_PURPLE_08, linewidth=0.8)
        ax.axvline(np.mean(ars), color='#e74c3c', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(ars):.2f}')
        ax.set_xlabel("Aspect Ratio (W/H)", fontsize=11)
        ax.set_ylabel("Frequency",          fontsize=11)
        ax.set_title("Aspect Ratio Distribution", fontsize=13, fontweight='bold', color='#2ed573', pad=12)
        ax.legend(facecolor='#0d1f0d', edgecolor=_GREEN_03)
        ax.grid(True, alpha=0.15)
    plt.tight_layout()
    return fig

def create_detection_density_chart(detections, image_shape):
    dark_chart_style()
    h, w = image_shape[:2]
    grid_cols, grid_rows = 5, 5
    cell_w, cell_h = w//grid_cols, h//grid_rows
    grid = np.zeros((grid_rows, grid_cols))
    for det in detections:
        col = min(det["CenterX"]//cell_w, grid_cols-1)
        row = min(det["CenterY"]//cell_h, grid_rows-1)
        grid[row, col] += 1
    fig, ax = plt.subplots(figsize=(8,7), facecolor='#0a140a')
    ax.set_facecolor('#0d1f0d')
    im = ax.imshow(grid, cmap='YlOrRd', interpolation='nearest')
    for i in range(grid_rows):
        for j in range(grid_cols):
            ax.text(j, i, f'{int(grid[i,j])}', ha='center', va='center',
                    color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel("Image Width Zones",  fontsize=10)
    ax.set_ylabel("Image Height Zones", fontsize=10)
    ax.set_title("Detection Density Grid (5×5)", fontsize=13, fontweight='bold', color='#2ed573', pad=12)
    plt.colorbar(im, ax=ax, label="Detection Count")
    plt.tight_layout()
    return fig

def create_roc_curve(detections):
    dark_chart_style()
    confidences = sorted([float(d["Confidence"]) for d in detections], reverse=True)
    thresholds  = np.linspace(0, 1, 100)
    tpr, fpr = [], []
    for thresh in thresholds:
        detected = sum(1 for c in confidences if c >= thresh)
        tpr.append(detected / max(len(confidences), 1))
        fpr.append(1 - tpr[-1])
    auc = np.trapz(tpr, fpr)
    fig, ax = plt.subplots(figsize=(8,6), facecolor='#0a140a')
    ax.set_facecolor('#0d1f0d')
    ax.plot(fpr, tpr, color='#2ed573', linewidth=2.5, label=f'ROC (AUC≈{abs(auc):.3f})')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#2ed573')
    ax.plot([0,1],[0,1], '#e74c3c', linestyle='--', linewidth=2, label='Random Classifier')
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("ROC Curve Analysis", fontsize=13, fontweight='bold', color='#2ed573', pad=12)
    ax.legend(facecolor='#0d1f0d', edgecolor=_GREEN_03)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    return fig

def create_class_statistics(detections):
    if not detections:
        return pd.DataFrame()
    df    = pd.DataFrame(detections)
    stats = []
    for cls in df['Class'].unique():
        cd   = df[df['Class'] == cls]
        conf = cd['Confidence'].astype(float)
        stats.append({
            'Class':          cls,
            'Count':          len(cd),
            'Avg Confidence': f"{conf.mean():.4f}",
            'Min Confidence': f"{conf.min():.4f}",
            'Max Confidence': f"{conf.max():.4f}",
            'Avg Area':       f"{cd['Area'].mean():.0f}",
            'Avg Width':      f"{cd['Width'].mean():.0f}",
            'Avg Height':     f"{cd['Height'].mean():.0f}",
            'Avg Aspect Ratio': f"{cd['AspectRatio'].astype(float).mean():.2f}"
        })
    return pd.DataFrame(stats)

def calculate_health_score(detections, image_shape):
    h, w = image_shape[:2]
    if not detections:
        return 0
    detected_area = sum(d["Area"] for d in detections)
    coverage      = (detected_area / (h * w)) * 100
    avg_conf      = np.mean([float(d["Confidence"]) for d in detections])
    count_score   = min((len(detections) / max((h*w/10000), 1)) * 100, 100)
    return min(coverage*0.3 + avg_conf*100*0.4 + count_score*0.3, 100)

# ==================== INSIGHT GENERATORS ====================

def _plain(text):
    """Wrap a plain-English explanation."""
    return f'<div class="plain-explanation">💡 <b>In plain terms:</b> {text}</div>'

def get_heatmap_insight(detections, heatmap_type="density"):
    if not detections:
        return "No detections available for heatmap analysis.", ""
    n   = len(detections)
    cx  = [d["CenterX"] for d in detections]
    cy  = [d["CenterY"] for d in detections]

    if heatmap_type == "density":
        sx, sy  = (np.std(cx) if len(cx)>1 else 0), (np.std(cy) if len(cy)>1 else 0)
        cluster = "tightly clustered" if (sx<100 and sy<100) else "widely spread out"
        zone    = "left" if np.mean(cx)<0.33 else ("right" if np.mean(cx)>0.66 else "center")
        technical = (f"**Density Heatmap:** {n} object(s) detected, {cluster}. "
                     f"Hotspot toward the **{zone}** side. σX={sx:.1f}px, σY={sy:.1f}px.")
        plain = (f"The bright areas show where your crops are — there {'are a lot of crops bunched together' if 'tightly' in cluster else 'are crops spread across the field'}. "
                 f"Most activity is on the {zone} side of the image.")
        return technical, plain

    elif heatmap_type == "proximity":
        if n < 2:
            return "Need at least 2 detections for proximity analysis.", ""
        dists = [np.sqrt((d2["CenterX"]-d1["CenterX"])**2 + (d2["CenterY"]-d1["CenterY"])**2)
                 for i,d1 in enumerate(detections) for d2 in detections[i+1:]]
        avg_d = np.mean(dists)
        technical = (f"**Proximity Map:** {n*(n-1)//2} pair(s) analyzed. "
                     f"Avg distance: **{avg_d:.1f}px**. "
                     f"Range: {min(dists):.1f}–{max(dists):.1f}px.")
        plain = (f"This shows how far apart your crops are from each other. "
                 f"On average they are {avg_d:.0f} pixels apart — "
                 f"{'they are growing very close together (dense planting).' if avg_d<150 else 'they are well spaced out (normal planting density).'}")
        return technical, plain
    return "Analysis complete.", ""

def get_gradcam_insight(detections):
    if not detections:
        return "No detections for Grad-CAM analysis.", ""
    confs    = [float(d["Confidence"]) for d in detections]
    avg_conf = np.mean(confs)
    high_n   = sum(1 for c in confs if c > 0.8)
    technical = (f"**Grad-CAM Attention:** Avg activation confidence: **{avg_conf:.3f}**. "
                 f"{high_n} high-confidence (>0.80) detection(s). "
                 f"{'Model attention is well-localised.' if avg_conf>0.7 else 'Consider adjusting confidence threshold.'}")
    plain = (f"The colored overlay shows the areas the AI was most interested in when looking at your image. "
             f"Red/hot areas = the AI is very confident about a crop there. "
             f"{'Overall the AI looks very certain — great image quality!' if avg_conf>0.7 else 'Some areas have lower certainty — try a clearer image if possible.'}")
    return technical, plain

def get_3d_insight(detections):
    if not detections:
        return "No 3D data to analyze.", ""
    classes = list(set(d["Class"] for d in detections))
    areas   = [d["Area"] for d in detections]
    confs   = [float(d["Confidence"]) for d in detections]
    technical = (f"**3D View:** {len(detections)} detection(s) across {len(classes)} class(es): {', '.join(classes)}. "
                 f"Confidence range: {min(confs):.3f}–{max(confs):.3f}. "
                 f"Area range: {min(areas)}–{max(areas)} px².")
    plain = (f"This 3D chart shows each detected crop as a dot. "
             f"Left/right & front/back = where in the image it is. "
             f"Height = how confident the AI is. Tall dots = very confident detections. "
             f"{'Multiple crop types were found.' if len(classes)>1 else 'Only one crop type was found in this image.'}")
    return technical, plain

def get_analytics_insight(detections):
    if not detections:
        return "No detections for analytics.", ""
    confs  = [float(d["Confidence"]) for d in detections]
    areas  = [d["Area"] for d in detections]
    cc     = {}
    for d in detections:
        cc[d["Class"]] = cc.get(d["Class"], 0) + 1
    dominant = max(cc, key=cc.get) if cc else "N/A"
    technical = (f"**Analytics:** {len(detections)} total detection(s). "
                 f"Dominant: **{dominant}** ({cc.get(dominant,0)} instances). "
                 f"Confidence — Mean: {np.mean(confs):.3f}, Std: {np.std(confs):.3f}. "
                 f"Avg size: {np.mean(areas):.0f} px².")
    plain = (f"Your image has {len(detections)} crops detected in total. "
             f"The most common crop is {dominant} ({cc.get(dominant,0)} found). "
             f"The AI is {'very confident' if np.mean(confs)>0.75 else 'moderately confident' if np.mean(confs)>0.5 else 'less certain'} about these detections "
             f"(average score: {np.mean(confs):.0%}).")
    return technical, plain

def get_image_analysis_insight(image_quality, detections):
    blur       = float(image_quality["Blur Score"])
    brightness = float(image_quality["Brightness"])
    quality    = image_quality["Image Quality"]
    technical  = (f"**Image Quality:** Rated **{quality}**. "
                  f"Sharpness: {blur:.1f} ({'excellent' if blur>200 else 'adequate' if blur>100 else 'low — may limit accuracy'}). "
                  f"Brightness: {brightness:.0f}/255 ({'well-lit' if 80<brightness<200 else 'may be over/under-exposed'}).")
    plain = (f"Your image quality is '{quality}'. "
             f"{'The photo is nice and sharp — perfect for detection!' if blur>100 else 'The photo is a bit blurry, which may reduce accuracy. Try a clearer image next time.'} "
             f"{'Lighting looks good.' if 80<brightness<200 else 'Lighting is not ideal (too dark or too bright) — this can affect results.'}")
    return technical, plain

def get_advanced_charts_insight(detections):
    if not detections:
        return "No detections for advanced analysis.", ""
    ws = [d["Width"]  for d in detections]
    hs = [d["Height"] for d in detections]
    ar = [float(d["AspectRatio"]) for d in detections]
    technical = (f"**Advanced Charts:** {len(detections)} detection(s). "
                 f"Width: {min(ws)}–{max(ws)}px (mean {np.mean(ws):.1f}). "
                 f"Height: {min(hs)}–{max(hs)}px (mean {np.mean(hs):.1f}). "
                 f"Avg aspect ratio: {np.mean(ar):.2f}.")
    plain = (f"These charts show the size and shape of each detected crop box. "
             f"On average, detected objects are {np.mean(ws):.0f}px wide and {np.mean(hs):.0f}px tall. "
             f"{'They are roughly square-shaped.' if 0.8<np.mean(ar)<1.2 else 'They tend to be wider than they are tall.' if np.mean(ar)>1.2 else 'They tend to be taller than wide.'}")
    return technical, plain

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
        fontSize=9.5, textColor=rl_colors.HexColor('#222222'), leading=15, spaceAfter=4)
    insight_style = ParagraphStyle('Insight', parent=styles['Normal'],
        fontSize=9, textColor=rl_colors.HexColor('#1a3a2a'), leading=14,
        leftIndent=12, rightIndent=12,
        backColor=rl_colors.HexColor('#e8f5ee'), borderPadding=(6,8,6,8))
    return styles, title_style, h2_style, h3_style, body_style, insight_style

def generate_pdf_report(image_pil, annotated_img, detections, image_quality, health_score, stats_df, section="full"):
    try:
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4,
                                leftMargin=0.75*inch, rightMargin=0.75*inch,
                                topMargin=0.75*inch, bottomMargin=0.75*inch)
        styles, title_style, h2_style, h3_style, body_style, insight_style = make_pdf_styles()
        story = []

        story.append(Paragraph("AgroVision", title_style))
        story.append(Paragraph("SSL & Graph-Refined Object Detection for Precision Agriculture",
            ParagraphStyle('sub', parent=styles['Normal'], fontSize=11, alignment=1,
                           textColor=rl_colors.HexColor('#4a8c5c'))))
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

        metrics_data = [
            ["Metric","Value","Metric","Value"],
            ["Health Score", f"{health_score:.1f}/100", "Total Detections", str(len(detections))],
            ["Avg Confidence",
             f"{np.mean([float(d['Confidence']) for d in detections]):.4f}" if detections else "N/A",
             "Image Quality", image_quality["Image Quality"]],
            ["Blur Score", image_quality["Blur Score"], "Brightness", image_quality["Brightness"]],
        ]
        mt = Table(metrics_data, colWidths=[1.5*inch]*4)
        mt.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),rl_colors.HexColor('#1a5c2e')),
            ('TEXTCOLOR',(0,0),(-1,0),rl_colors.white),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('BACKGROUND',(0,1),(0,-1),rl_colors.HexColor('#e8f5ee')),
            ('BACKGROUND',(2,1),(2,-1),rl_colors.HexColor('#e8f5ee')),
            ('FONTNAME',(0,1),(0,-1),'Helvetica-Bold'),
            ('FONTNAME',(2,1),(2,-1),'Helvetica-Bold'),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('FONTSIZE',(0,0),(-1,-1),9),
            ('GRID',(0,0),(-1,-1),0.5,rl_colors.HexColor('#aaddbb')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[rl_colors.white,rl_colors.HexColor('#f0faf3')]),
            ('BOTTOMPADDING',(0,0),(-1,-1),7),
            ('TOPPADDING',(0,0),(-1,-1),7),
        ]))
        story.append(mt)
        story.append(Spacer(1, 0.2*inch))

        if section in ("full","detection"):
            story.append(HRFlowable(width="100%",thickness=1,color=rl_colors.HexColor('#aaddbb')))
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph("Detection Results", h2_style))
            orig_buf = io.BytesIO(); image_pil.save(orig_buf, format='PNG'); orig_buf.seek(0)
            ann_buf  = io.BytesIO(); annotated_img.save(ann_buf, format='PNG'); ann_buf.seek(0)
            img_table = Table([
                [Paragraph("Original Image", h3_style), Paragraph("Detected Image", h3_style)],
                [RLImage(orig_buf,width=3*inch,height=2.2*inch), RLImage(ann_buf,width=3*inch,height=2.2*inch)]
            ], colWidths=[3.5*inch, 3.5*inch])
            img_table.setStyle(TableStyle([
                ('ALIGN',(0,0),(-1,-1),'CENTER'),
                ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ]))
            story.append(img_table)
            story.append(Spacer(1, 0.15*inch))
            if detections:
                story.append(Paragraph("Insight", h3_style))
                story.append(Paragraph(
                    f"Found {len(detections)} crop(s). "
                    f"Average confidence: {np.mean([float(d['Confidence']) for d in detections]):.3f}. "
                    f"Health score: {health_score:.1f}/100 — "
                    f"{'Excellent crop coverage' if health_score>75 else 'Moderate coverage' if health_score>40 else 'Low coverage detected'}.",
                    insight_style))
            story.append(Spacer(1, 0.15*inch))

        if section in ("full","detection","analytics") and not stats_df.empty:
            story.append(HRFlowable(width="100%",thickness=1,color=rl_colors.HexColor('#aaddbb')))
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph("Class-wise Statistics", h2_style))
            header = list(stats_df.columns)
            data   = [header] + [list(row) for _,row in stats_df.iterrows()]
            tbl    = Table(data, repeatRows=1)
            tbl.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),rl_colors.HexColor('#1a5c2e')),
                ('TEXTCOLOR',(0,0),(-1,0),rl_colors.white),
                ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                ('ALIGN',(0,0),(-1,-1),'CENTER'),
                ('FONTSIZE',(0,0),(-1,-1),8.5),
                ('GRID',(0,0),(-1,-1),0.5,rl_colors.HexColor('#aaddbb')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[rl_colors.white,rl_colors.HexColor('#f0faf3')]),
                ('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 0.15*inch))
            tech, _ = get_analytics_insight(detections)
            story.append(Paragraph("Analytics Insight", h3_style))
            story.append(Paragraph(tech.replace("**",""), insight_style))

        if section in ("full","detection") and detections:
            story.append(PageBreak())
            story.append(Paragraph("All Detections", h2_style))
            det_df     = pd.DataFrame(detections)
            det_header = list(det_df.columns)
            det_data   = [det_header] + [list(row) for _,row in det_df.iterrows()]
            det_tbl    = Table(det_data, repeatRows=1)
            det_tbl.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),rl_colors.HexColor('#1a5c2e')),
                ('TEXTCOLOR',(0,0),(-1,0),rl_colors.white),
                ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                ('ALIGN',(0,0),(-1,-1),'CENTER'),
                ('FONTSIZE',(0,0),(-1,-1),7.5),
                ('GRID',(0,0),(-1,-1),0.5,rl_colors.HexColor('#aaddbb')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[rl_colors.white,rl_colors.HexColor('#f0faf3')]),
                ('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),
            ]))
            story.append(det_tbl)

        doc.build(story)
        pdf_buffer.seek(0)
        return pdf_buffer
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

def generate_section_pdf(section_name, insight_text, fig=None,
                          detections=None, image_quality=None, health_score=0, stats_df=None):
    try:
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4,
                                leftMargin=0.75*inch, rightMargin=0.75*inch,
                                topMargin=0.75*inch, bottomMargin=0.75*inch)
        styles, title_style, h2_style, h3_style, body_style, insight_style = make_pdf_styles()
        story = []

        story.append(Paragraph("AgroVision", title_style))
        story.append(HRFlowable(width="100%",thickness=2,color=rl_colors.HexColor('#2ed573')))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(f"{section_name} Report", h2_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 0.15*inch))

        if image_quality:
            n = len(detections) if detections else 0
            metrics_data = [
                ["Metric","Value","Metric","Value"],
                ["Health Score",f"{health_score:.1f}/100","Detections",str(n)],
                ["Image Quality",image_quality.get("Image Quality","N/A"),"Avg Confidence",
                 f"{np.mean([float(d['Confidence']) for d in detections]):.4f}" if detections else "N/A"],
            ]
            mt = Table(metrics_data, colWidths=[1.5*inch]*4)
            mt.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),rl_colors.HexColor('#1a5c2e')),
                ('TEXTCOLOR',(0,0),(-1,0),rl_colors.white),
                ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                ('ALIGN',(0,0),(-1,-1),'CENTER'),
                ('FONTSIZE',(0,0),(-1,-1),9),
                ('GRID',(0,0),(-1,-1),0.5,rl_colors.HexColor('#aaddbb')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[rl_colors.white,rl_colors.HexColor('#f0faf3')]),
                ('BOTTOMPADDING',(0,0),(-1,-1),7),
            ]))
            story.append(mt)
            story.append(Spacer(1, 0.15*inch))

        if fig:
            story.append(Paragraph("Visualization", h3_style))
            story.append(RLImage(fig_to_bytes(fig), width=5.5*inch, height=3.5*inch))
            story.append(Spacer(1, 0.15*inch))

        story.append(Paragraph("AI Analysis & Insights", h3_style))
        story.append(Paragraph(insight_text.replace("**","").replace("*",""), insight_style))

        doc.build(story)
        pdf_buffer.seek(0)
        return pdf_buffer
    except Exception as e:
        st.error(f"PDF error: {e}")
        return None

def export_detections_json(detections, image_filename):
    return json.dumps({
        "timestamp": datetime.now().isoformat(),
        "image_filename": image_filename,
        "total_detections": len(detections),
        "detections": detections
    }, indent=2)

# ==================== INSIGHT BOX HELPER ====================

def render_insight_and_pdf(technical_text, plain_text, section_name, icon,
                            pdf_fig=None, detections=None, image_quality=None, health_score=0,
                            key_suffix=""):
    # Render insight box
    safe_tech = technical_text.replace("**", "<b>", 1)
    # crude bold replacement
    result = ""
    toggle = False
    for ch in technical_text:
        if ch == '*':
            if not toggle:
                result += '<b>'
                toggle = True
            else:
                result += '</b>'
                toggle = False
        else:
            result += ch
    st.markdown(f"""
    <div class="analysis-insight">
        <div class="insight-title">{icon} AI Analysis Insight</div>
        <span>{result}</span>
        {_plain(plain_text) if plain_text else ""}
    </div>
    """, unsafe_allow_html=True)

    # Section PDF button
    col_pad, col_btn = st.columns([4, 1])
    with col_btn:
        key = f"pdf_{section_name}_{key_suffix}"
        if st.button(f"📄 PDF", key=key):
            with st.spinner(f"Generating {section_name} PDF..."):
                iq = image_quality or {}
                pdf_buf = generate_section_pdf(
                    section_name, technical_text, pdf_fig,
                    detections, iq, health_score
                )
                if pdf_buf:
                    st.download_button(
                        f"⬇️ Download PDF",
                        data=pdf_buf,
                        file_name=f"agrovision_{section_name.lower().replace(' ','_')}.pdf",
                        mime="application/pdf",
                        key=f"dl_{section_name}_{key_suffix}"
                    )

# ==================== PER-IMAGE ANALYSIS ====================

def analyze_single_image(uploaded_file, model, conf_threshold, iou_threshold,
                          filter_class, min_conf_filter, max_conf_filter,
                          min_size, max_size,
                          enable_gradcam, enable_heatmap, enable_3d,
                          enable_preprocessing, enable_proximity, enable_grid,
                          img_index=0):
    """Run full analysis for one uploaded image and render all tabs."""
    image_pil = Image.open(uploaded_file).convert('RGB')
    image_np  = pil_to_cv2(image_pil)

    results = run_detection(image_np, model, conf_threshold, iou_threshold)
    annotated_img, detections = draw_detections_advanced(image_np, results, conf_threshold, show_grid=enable_grid)
    filtered_detections = filter_detections(detections, filter_class, min_conf_filter, max_conf_filter, min_size, max_size)

    image_quality = analyze_image_quality(image_np)
    health_score  = calculate_health_score(filtered_detections, image_np.shape)
    stats_df      = create_class_statistics(filtered_detections)
    avg_conf      = np.mean([float(d["Confidence"]) for d in filtered_detections]) if filtered_detections else 0
    ks            = str(img_index)  # unique key suffix per image

    # ── Key Metrics Row ──
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("🌾 Health Score",  f"{health_score:.1f}/100")
    with c2: st.metric("📊 Detections",    len(filtered_detections))
    with c3: st.metric("📈 Avg Confidence",f"{avg_conf:.4f}")
    with c4: st.metric("🔍 Image Quality", image_quality["Image Quality"])

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        ["📸 Detection","🔥 Grad-CAM","🌡️ Heatmaps","3️⃣ 3D View",
         "🔬 Image Analysis","📊 Analytics","📈 Advanced Charts","📋 Details"]
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

        tech = (f"**Detection Result:** Found {len(filtered_detections)} object(s) with avg confidence {avg_conf:.3f}. "
                f"Health score: {health_score:.1f}/100.")
        plain = (f"The AI found {len(filtered_detections)} crop(s) in your image. "
                 f"{'That is excellent coverage — your field looks healthy!' if health_score>75 else 'Moderate number of crops detected.' if health_score>40 else 'Very few crops detected — try lowering the confidence threshold in the sidebar.'}")
        render_insight_and_pdf(tech, plain, "Detection", "📸",
                               detections=filtered_detections, image_quality=image_quality,
                               health_score=health_score, key_suffix=ks)

    # ── TAB 2: Grad-CAM ──
    with tab2:
        if enable_gradcam:
            st.markdown('<div class="section-header">🔥 Grad-CAM Attention Heatmap</div>', unsafe_allow_html=True)
            try:
                gradcam_img, heatmap = generate_gradcam(image_np, model)
                if gradcam_img:
                    col1, col2 = st.columns(2)
                    with col1: st.image(gradcam_img, caption="Overlay", use_column_width=True)
                    with col2: st.image(Image.fromarray((heatmap*255).astype(np.uint8)),
                                        caption="Raw Heatmap", use_column_width=True, channels="GRAY")
                    tech, plain = get_gradcam_insight(filtered_detections)
                    render_insight_and_pdf(tech, plain, "Grad-CAM", "🔥",
                                           detections=filtered_detections, image_quality=image_quality,
                                           health_score=health_score, key_suffix=ks)
            except Exception as e:
                st.error(f"Grad-CAM error: {e}")
        else:
            st.info("Grad-CAM disabled in settings")

    # ── TAB 3: Heatmaps ──
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            if enable_heatmap:
                st.markdown('<div class="section-header">Density Heatmap</div>', unsafe_allow_html=True)
                density_img, _ = generate_density_heatmap(filtered_detections, image_np.shape)
                st.image(density_img, use_column_width=True)
        with col2:
            if enable_proximity:
                st.markdown('<div class="section-header">Proximity Map</div>', unsafe_allow_html=True)
                prox_img = create_proximity_map(filtered_detections, image_np.shape)
                if prox_img:
                    st.image(prox_img, use_column_width=True)
                else:
                    st.info("Need 2+ detections for proximity map")

        dtech, dplain = get_heatmap_insight(filtered_detections, "density")
        ptech, pplain = get_heatmap_insight(filtered_detections, "proximity")
        combined_tech  = dtech + " | " + ptech
        combined_plain = dplain + " " + pplain
        render_insight_and_pdf(combined_tech, combined_plain, "Heatmap", "🌡️",
                               detections=filtered_detections, image_quality=image_quality,
                               health_score=health_score, key_suffix=ks)

    # ── TAB 4: 3D View ──
    with tab4:
        if enable_3d:
            st.markdown('<div class="section-header">🔮 3D Detection Visualization</div>', unsafe_allow_html=True)
            fig_3d = create_3d_plot(filtered_detections)
            if fig_3d:
                st.plotly_chart(fig_3d, use_container_width=True)
                tech, plain = get_3d_insight(filtered_detections)
                render_insight_and_pdf(tech, plain, "3D Visualization", "🔮",
                                       detections=filtered_detections, image_quality=image_quality,
                                       health_score=health_score, key_suffix=ks)
            else:
                st.info("No detections to visualize in 3D")
        else:
            st.info("3D visualization disabled")

    # ── TAB 5: Image Analysis ──
    with tab5:
        if enable_preprocessing:
            col1, col2 = st.columns(2)
            color_fig = None
            with col1:
                st.markdown('<div class="section-header">Image Quality Metrics</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame([image_quality]), use_container_width=True)
                st.markdown('<div class="section-header">Edge Detection</div>', unsafe_allow_html=True)
                st.image(create_edge_detection(image_np), caption="Canny Edges", use_column_width=True, channels="GRAY")
            with col2:
                st.markdown('<div class="section-header">Histogram Equalization</div>', unsafe_allow_html=True)
                st.image(create_histogram_equalized(image_np), use_column_width=True)
                st.markdown('<div class="section-header">Color Channel Distribution</div>', unsafe_allow_html=True)
                color_fig = create_color_distribution(image_np)
                st.pyplot(color_fig, use_container_width=True)
            tech, plain = get_image_analysis_insight(image_quality, filtered_detections)
            render_insight_and_pdf(tech, plain, "Image Analysis", "🔬",
                                   pdf_fig=color_fig,
                                   detections=filtered_detections, image_quality=image_quality,
                                   health_score=health_score, key_suffix=ks)
        else:
            st.info("Image Analysis disabled in settings")

    # ── TAB 6: Analytics ──
    with tab6:
        col1, col2 = st.columns(2)
        class_fig = None
        with col1:
            st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
            class_fig = create_class_distribution(filtered_detections)
            st.pyplot(class_fig, use_container_width=True)
        with col2:
            st.markdown('<div class="section-header">Confidence Distribution</div>', unsafe_allow_html=True)
            conf_fig = create_confidence_distribution(filtered_detections)
            st.pyplot(conf_fig, use_container_width=True)
        tech, plain = get_analytics_insight(filtered_detections)
        render_insight_and_pdf(tech, plain, "Analytics", "📊",
                               pdf_fig=class_fig,
                               detections=filtered_detections, image_quality=image_quality,
                               health_score=health_score, key_suffix=ks)

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
            st.pyplot(create_aspect_ratio_distribution(filtered_detections), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">Detection Density Grid</div>', unsafe_allow_html=True)
            st.pyplot(create_detection_density_chart(filtered_detections, image_np.shape), use_container_width=True)
        with col2:
            st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
            st.pyplot(create_roc_curve(filtered_detections), use_container_width=True)
        st.markdown('<div class="section-header">Bounding Box Size Distributions</div>', unsafe_allow_html=True)
        st.pyplot(create_bbox_distribution(filtered_detections), use_container_width=True)
        tech, plain = get_advanced_charts_insight(filtered_detections)
        render_insight_and_pdf(tech, plain, "Advanced Charts", "📈",
                               pdf_fig=scatter_fig,
                               detections=filtered_detections, image_quality=image_quality,
                               health_score=health_score, key_suffix=ks)

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
            st.dataframe(pd.DataFrame(filtered_detections), use_container_width=True)
        details_tech = (f"**Details:** {len(filtered_detections)} detection(s) across "
            f"{len(set(d['Class'] for d in filtered_detections)) if filtered_detections else 0} class(es). "
            f"Health Score: {health_score:.1f}/100. "
            f"Image quality: {image_quality['Image Quality']}.")
        details_plain = (f"This table lists every single crop the AI found. "
            f"You can see exactly where each one is (X1,Y1 to X2,Y2 coordinates), "
            f"how confident the AI was, and the size of each detection box.")
        render_insight_and_pdf(details_tech, details_plain, "Details", "📋",
                               detections=filtered_detections, image_quality=image_quality,
                               health_score=health_score, key_suffix=ks)

    # ── Export Section ──
    st.markdown("---")
    st.markdown('<div class="section-header">💾 Export & Download</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        img_buf = io.BytesIO()
        annotated_img.save(img_buf, format='PNG'); img_buf.seek(0)
        st.download_button("📸 Annotated Image", data=img_buf,
                           file_name=f"agrovision_detected_{uploaded_file.name}",
                           mime="image/png", use_container_width=True, key=f"dl_img_{ks}")
    with c2:
        if filtered_detections:
            st.download_button("📊 Detections CSV",
                               data=pd.DataFrame(filtered_detections).to_csv(index=False),
                               file_name=f"agrovision_detections_{img_index+1}.csv",
                               mime="text/csv", use_container_width=True, key=f"dl_csv_{ks}")
    with c3:
        st.download_button("📄 Detections JSON",
                           data=export_detections_json(filtered_detections, uploaded_file.name),
                           file_name=f"agrovision_detections_{img_index+1}.json",
                           mime="application/json", use_container_width=True, key=f"dl_json_{ks}")
    with c4:
        if st.button("📑 Full PDF Report", use_container_width=True, key=f"btn_pdf_{ks}"):
            with st.spinner("Generating PDF..."):
                pdf_buf = generate_pdf_report(image_pil, annotated_img, filtered_detections,
                                              image_quality, health_score, stats_df, section="full")
                if pdf_buf:
                    st.download_button("⬇️ Download Full PDF", data=pdf_buf,
                                       file_name=f"agrovision_report_{img_index+1}.pdf",
                                       mime="application/pdf", key=f"dl_pdf_{ks}")

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
            Use the sidebar to upload a <code>.pt</code> or <code>.pkl</code> model file
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    model, model_type = load_active_model(uploaded_models, selected_model_name)

    if model is not None and model_type == "yolo":

        # ── Image Upload (always multi) ──
        st.markdown('<div class="section-header">📁 Upload Crop Images</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload one or more images — full analysis will be shown for each",
            type=["jpg","jpeg","png","bmp"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if uploaded_files:
            n_imgs = len(uploaded_files)

            # ── Batch summary bar ──
            if n_imgs > 1:
                st.markdown(f"""
                <div style='background:rgba(46,213,115,0.08);border:1px solid rgba(46,213,115,0.2);
                            border-radius:12px;padding:0.9rem 1.3rem;margin-bottom:1rem;
                            display:flex;align-items:center;gap:12px;'>
                    <span style='font-size:1.4rem;'>🗂️</span>
                    <span style='color:rgba(200,255,200,0.8);font-size:0.95rem;font-weight:500;'>
                        <b style='color:#2ed573;'>{n_imgs} images</b> uploaded — 
                        a separate full report is shown below for each image.
                    </span>
                </div>
                """, unsafe_allow_html=True)

            # ── Per-image analysis ──
            for idx, uploaded_file in enumerate(uploaded_files):
                header = f"📸 Image {idx+1} of {n_imgs}: {uploaded_file.name}"
                with st.expander(header, expanded=(idx == 0)):
                    analyze_single_image(
                        uploaded_file, model,
                        conf_threshold, iou_threshold,
                        filter_class, min_conf_filter, max_conf_filter,
                        min_size, max_size,
                        enable_gradcam, enable_heatmap, enable_3d,
                        enable_preprocessing, enable_proximity, enable_grid,
                        img_index=idx
                    )

            # ── Batch ZIP export when multiple images ──
            if n_imgs > 1:
                st.markdown("---")
                st.markdown('<div class="section-header">📦 Batch Export All Images</div>', unsafe_allow_html=True)
                if st.button("📦 Download All Results as ZIP", use_container_width=False):
                    with st.spinner("Packaging results…"):
                        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
                            with zipfile.ZipFile(tmp_zip, 'w') as z:
                                for idx2, uf in enumerate(uploaded_files):
                                    img_pil2 = Image.open(uf).convert('RGB')
                                    img_np2  = pil_to_cv2(img_pil2)
                                    res2     = run_detection(img_np2, model, conf_threshold, iou_threshold)
                                    ann2, det2 = draw_detections_advanced(img_np2, res2, conf_threshold)
                                    fdet2    = filter_detections(det2, filter_class, min_conf_filter,
                                                                 max_conf_filter, min_size, max_size)
                                    buf2 = io.BytesIO(); ann2.save(buf2, format='PNG')
                                    z.writestr(f"annotated_{uf.name}", buf2.getvalue())
                                    z.writestr(f"detections_{uf.name}.json",
                                               export_detections_json(fdet2, uf.name))
                            with open(tmp_zip.name, 'rb') as fz:
                                st.download_button("⬇️ Download ZIP", data=fz.read(),
                                                   file_name="agrovision_batch.zip",
                                                   mime="application/zip")

        else:
            st.markdown("""
            <div style='background:rgba(52,152,219,0.08);border:1px solid rgba(52,152,219,0.2);
                        border-radius:14px;padding:1.5rem;text-align:center;margin-top:1rem;'>
                <div style='font-size:2rem;margin-bottom:0.5rem;'>👆</div>
                <div style='color:rgba(200,220,255,0.7);font-size:0.95rem;'>
                    Upload one or more crop images above to begin analysis
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif model is not None and model_type == "pkl":
        st.info("🧪 PKL model loaded. Integrate your custom prediction logic here.")
        st.code(f"Model type: {type(model).__name__}", language="python")

    elif model is None:
        st.error("⚠️ Failed to load the selected model. Please check the file format.")

# ── Footer ──
st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:1rem 0;'>
    <span style='color:rgba(46,213,115,0.4);font-size:0.8rem;letter-spacing:2px;'>
        🌿 AGROVISION · SSL & GRAPH-REFINED OBJECT DETECTION · PRECISION AGRICULTURE AI
    </span>
</div>
""", unsafe_allow_html=True)