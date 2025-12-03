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
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# Set page config
st.set_page_config(page_title="Advanced Crop Detection", layout="wide", initial_sidebar_state="expanded")
st.title("🌾 Advanced Crop Detection System - YOLOv11")
st.markdown("Professional-grade Sunflower & Rice Detection with AI Insights")

# Initialize session state
if 'roi_mode' not in st.session_state:
    st.session_state.roi_mode = False
if 'roi_points' not in st.session_state:
    st.session_state.roi_points = []

# ==================== SIDEBAR CONFIGURATION ====================
with st.sidebar:
    st.header("⚙️ Advanced Configuration")
    
    # Model upload
    uploaded_model = st.file_uploader(
        "📤 Upload best.pt Model",
        type=["pt"],
        help="Upload your trained YOLOv11 best.pt weights"
    )
    
    # Thresholds
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    iou_threshold = st.slider(
        "IOU Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.45,
        step=0.05
    )
    
    st.markdown("---")
    st.subheader("📊 Visualization Settings")
    
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
    st.subheader("🎯 Filtering Options")
    
    filter_class = st.multiselect(
        "Filter by Class",
        ["All", "Sunflower", "Rice"],
        default=["All"]
    )
    
    min_conf_filter = st.slider(
        "Min Confidence Filter",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05
    )
    
    max_conf_filter = st.slider(
        "Max Confidence Filter",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05
    )
    
    min_size = st.number_input("Min Detection Area (px²)", value=0, min_value=0)
    max_size = st.number_input("Max Detection Area (px²)", value=10000000, min_value=0)
    
    st.markdown("---")
    st.subheader("📁 Batch Processing")
    
    batch_mode = st.checkbox("Enable Batch Mode", value=False)
    if batch_mode:
        st.info("Upload multiple images for batch analysis")

# ==================== HELPER FUNCTIONS ====================

@st.cache_resource
def load_model_from_file(model_file):
    try:
        with open("temp_model.pt", "wb") as f:
            f.write(model_file.getbuffer())
        model = YOLO("temp_model.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

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
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            cls_name = result.names[cls]
            
            cv2.rectangle(image_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image_annotated, (x1, y1 - 25), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image_annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(image_annotated, (cx, cy), 3, (0, 0, 255), -1)
            
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
            cv2.line(image_annotated, (0, i), (w, i), (100, 100, 100), 1)
        for i in range(0, w, grid_size):
            cv2.line(image_annotated, (i, 0), (i, h), (100, 100, 100), 1)
    
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
    
    fig = px.scatter_3d(
        df,
        x="CenterX",
        y="CenterY",
        z="Confidence_float",
        color="Class",
        size="Area",
        hover_data=["Confidence", "Area"],
        title="3D Detection Analysis",
        labels={"CenterX": "X Position", "CenterY": "Y Position", "Confidence_float": "Confidence"}
    )
    
    fig.update_layout(height=600, width=800)
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

def create_color_distribution(image_np):
    colors_cv = ('b', 'g', 'r')
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for i, col in enumerate(colors_cv):
        hist = cv2.calcHist([image_np], [i], None, [256], [0, 256])
        ax.plot(hist, color=col, label=['Blue', 'Green', 'Red'][i])
    
    ax.set_xlabel("Intensity", fontweight='bold')
    ax.set_ylabel("Frequency", fontweight='bold')
    ax.set_title("Color Distribution", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def create_confidence_distribution(detections):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    confidences = [float(d["Confidence"]) for d in detections]
    
    if confidences:
        ax.hist(confidences, bins=15, color='#45B7D1', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
        ax.axvline(np.median(confidences), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(confidences):.3f}')
        ax.set_xlabel("Confidence Score", fontsize=11, fontweight='bold')
        ax.set_ylabel("Frequency", fontsize=11, fontweight='bold')
        ax.set_title("Confidence Distribution", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_class_distribution(detections):
    class_counts = {}
    for det in detections:
        cls = det["Class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if class_counts:
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
        wedges, texts, autotexts = ax.pie(class_counts.values(), labels=class_counts.keys(), 
                                            autopct='%1.1f%%', colors=colors, startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.set_title("Class Distribution", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_scatter_plot(detections):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if detections:
        areas = [d["Area"] for d in detections]
        confidences = [float(d["Confidence"]) for d in detections]
        classes = [d["Class"] for d in detections]
        
        unique_classes = list(set(classes))
        colors_map = {cls: plt.cm.tab10(i) for i, cls in enumerate(unique_classes)}
        colors = [colors_map[cls] for cls in classes]
        
        scatter = ax.scatter(areas, confidences, s=200, c=colors, alpha=0.6, edgecolors='black', linewidth=1.5)
        
        ax.set_xlabel("Bounding Box Area (pixels)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Confidence Score", fontsize=11, fontweight='bold')
        ax.set_title("Confidence vs Bounding Box Size", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        legend_labels = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_map[cls], 
                                    markersize=8, label=cls) for cls in unique_classes]
        ax.legend(handles=legend_labels, loc='best')
    
    plt.tight_layout()
    return fig

def create_bbox_distribution(detections):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if detections:
        widths = [d["Width"] for d in detections]
        heights = [d["Height"] for d in detections]
        
        axes[0].hist(widths, bins=10, color='#FF6B6B', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel("Width (pixels)", fontweight='bold')
        axes[0].set_ylabel("Frequency", fontweight='bold')
        axes[0].set_title("Width Distribution", fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(heights, bins=10, color='#4ECDC4', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel("Height (pixels)", fontweight='bold')
        axes[1].set_ylabel("Frequency", fontweight='bold')
        axes[1].set_title("Height Distribution", fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_aspect_ratio_distribution(detections):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if detections:
        aspect_ratios = [float(d["AspectRatio"]) for d in detections]
        
        ax.hist(aspect_ratios, bins=15, color='#95E1D3', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(aspect_ratios), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(aspect_ratios):.2f}')
        ax.set_xlabel("Aspect Ratio (Width/Height)", fontweight='bold')
        ax.set_ylabel("Frequency", fontweight='bold')
        ax.set_title("Aspect Ratio Distribution", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_detection_density_chart(detections, image_shape):
    h, w = image_shape[:2]
    
    grid_cols, grid_rows = 5, 5
    cell_w, cell_h = w // grid_cols, h // grid_rows
    
    grid = np.zeros((grid_rows, grid_cols))
    
    for det in detections:
        cx, cy = det["CenterX"], det["CenterY"]
        col, row = min(cx // cell_w, grid_cols - 1), min(cy // cell_h, grid_rows - 1)
        grid[row, col] += 1
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(grid, cmap='YlOrRd', interpolation='nearest')
    
    ax.set_xlabel("Image Width", fontweight='bold')
    ax.set_ylabel("Image Height", fontweight='bold')
    ax.set_title("Detection Density Grid", fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label="Detection Count")
    
    plt.tight_layout()
    return fig

def create_roc_curve(detections):
    confidences = sorted([float(d["Confidence"]) for d in detections], reverse=True)
    
    thresholds = np.linspace(0, 1, 100)
    tpr = []
    fpr = []
    
    for thresh in thresholds:
        detected = sum(1 for c in confidences if c >= thresh)
        tpr.append(detected / max(len(confidences), 1))
        fpr.append(1 - tpr[-1])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label='ROC Curve')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
    ax.set_xlabel("False Positive Rate", fontweight='bold')
    ax.set_ylabel("True Positive Rate", fontweight='bold')
    ax.set_title("ROC Curve Analysis", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_class_statistics(detections):
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

def generate_pdf_report(image_pil, annotated_img, detections, image_quality, health_score, stats_df):
    try:
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            alignment=1
        )
        
        story.append(Paragraph("🌾 Crop Detection Report", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("<b>Health Score</b>", styles['Heading2']))
        story.append(Paragraph(f"Overall Crop Health: <b>{health_score:.1f}/100</b>", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("<b>Detection Summary</b>", styles['Heading2']))
        summary_data = [
            ["Total Detections", str(len(detections))],
            ["Average Confidence", f"{np.mean([float(d['Confidence']) for d in detections]):.4f}"],
            ["Coverage (%)", f"{(sum(d['Area'] for d in detections) / (detections[0].get('CenterX', 640) * 2 * detections[0].get('CenterY', 480) * 2)) * 100 if detections else 0:.2f}"]
        ]
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph("<b>Image Quality Metrics</b>", styles['Heading2']))
        quality_data = [
            ["Metric", "Value"],
            ["Blur Score", image_quality["Blur Score"]],
            ["Contrast", image_quality["Contrast"]],
            ["Brightness", image_quality["Brightness"]],
            ["Quality", image_quality["Image Quality"]]
        ]
        quality_table = Table(quality_data)
        quality_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(quality_table)
        story.append(PageBreak())
        
        story.append(Paragraph("<b>Class-wise Statistics</b>", styles['Heading2']))
        
        stats_data = [["Class", "Count", "Avg Confidence", "Avg Area"]]
        for idx, row in stats_df.iterrows():
            stats_data.append([row['Class'], str(row['Count']), row['Avg Confidence'], row['Avg Area']])
        
        stats_table = Table(stats_data)
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph("<b>Detected Image</b>", styles['Heading2']))
        img_buffer = io.BytesIO()
        annotated_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        rl_image = RLImage(img_buffer, width=6*inch, height=4*inch)
        story.append(rl_image)
        
        doc.build(story)
        pdf_buffer.seek(0)
        return pdf_buffer
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

def export_detections_json(detections, image_filename):
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "image_filename": image_filename,
        "total_detections": len(detections),
        "detections": detections
    }
    return json.dumps(export_data, indent=2)

# ==================== MAIN APP ====================

st.markdown("---")

if uploaded_model is None:
    st.warning("⚠️ Please upload a best.pt model file in the sidebar to start")
else:
    model = load_model_from_file(uploaded_model)
    
    if model is not None:
        if batch_mode:
            st.subheader("📁 Batch Processing Mode")
            uploaded_files = st.file_uploader(
                "Upload multiple images",
                type=["jpg", "jpeg", "png", "bmp"],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                batch_results = []
                progress_bar = st.progress(0)
                
                for idx, uploaded_file in enumerate(uploaded_files):
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
                
                st.subheader("📊 Batch Results Summary")
                batch_df = pd.DataFrame([{
                    "Filename": r["filename"],
                    "Detections": r["detections"],
                    "Avg Confidence": f"{r['avg_confidence']:.4f}"
                } for r in batch_results])
                
                st.dataframe(batch_df, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    total_dets = sum(r["detections"] for r in batch_results)
                    st.metric("Total Detections (Batch)", total_dets)
                with col2:
                    avg_conf = np.mean([r["avg_confidence"] for r in batch_results])
                    st.metric("Average Confidence (Batch)", f"{avg_conf:.4f}")
                
                st.markdown("---")
                st.subheader("💾 Batch Export")
                
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
                
                st.info("🔄 Running detection...")
                results = run_detection(image_np, model, conf_threshold, iou_threshold)
                
                annotated_img, detections = draw_detections_advanced(image_np, results, conf_threshold, show_grid=enable_grid)
                
                filtered_detections = filter_detections(detections, filter_class, min_conf_filter, max_conf_filter, min_size, max_size)
                
                st.success(f"✅ Detection Complete! Found {len(filtered_detections)} objects (after filtering)")
                
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
                
                tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
                    ["📸 Detection", "🔥 Grad-CAM", "🌡️ Heatmaps", "3️⃣ 3D View", 
                     "🔬 Image Analysis", "📊 Analytics", "📈 Advanced Charts", "📋 Details"]
                )
                
                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original")
                        st.image(image_pil, use_column_width=True)
                    with col2:
                        st.subheader("Detected")
                        st.image(annotated_img, use_column_width=True)
                
                with tab2:
                    if enable_gradcam:
                        st.subheader("Grad-CAM Attention Heatmap")
                        try:
                            gradcam_img, heatmap = generate_gradcam(image_np, model)
                            if gradcam_img:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(gradcam_img, caption="Overlay", use_column_width=True)
                                with col2:
                                    st.image(Image.fromarray((heatmap * 255).astype(np.uint8)), 
                                            caption="Raw Heatmap", use_column_width=True, channels="GRAY")
                        except Exception as e:
                            st.error(f"Grad-CAM error: {e}")
                    else:
                        st.info("Grad-CAM disabled in settings")
                
                with tab3:
                    col1, col2 = st.columns(2)
                    with col1:
                        if enable_heatmap:
                            st.subheader("Density Heatmap")
                            density_img, density_map = generate_density_heatmap(filtered_detections, image_np.shape)
                            st.image(density_img, use_column_width=True)
                    with col2:
                        if enable_proximity:
                            st.subheader("Proximity Map")
                            proximity_img = create_proximity_map(filtered_detections, image_np.shape)
                            if proximity_img:
                                st.image(proximity_img, use_column_width=True)
                            else:
                                st.info("Need 2+ detections for proximity analysis")
                
                with tab4:
                    if enable_3d:
                        st.subheader("3D Detection Visualization")
                        fig_3d = create_3d_plot(filtered_detections)
                        if fig_3d:
                            st.plotly_chart(fig_3d, use_container_width=True)
                        else:
                            st.info("No detections for 3D visualization")
                    else:
                        st.info("3D visualization disabled")
                
                with tab5:
                    if enable_preprocessing:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Image Quality")
                            quality_df = pd.DataFrame([image_quality])
                            st.dataframe(quality_df, use_container_width=True)
                            
                            st.subheader("Edge Detection")
                            edges = create_edge_detection(image_np)
                            st.image(edges, caption="Canny Edges", use_column_width=True, channels="GRAY")
                        
                        with col2:
                            st.subheader("Histogram Equalization")
                            equalized = create_histogram_equalized(image_np)
                            st.image(equalized, use_column_width=True)
                            
                            st.subheader("Color Distribution")
                            st.pyplot(create_color_distribution(image_np), use_container_width=True)
                    else:
                        st.info("Image analysis disabled")
                
                with tab6:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(create_class_distribution(filtered_detections), use_container_width=True)
                    with col2:
                        st.pyplot(create_confidence_distribution(filtered_detections), use_container_width=True)
                
                with tab7:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(create_scatter_plot(filtered_detections), use_container_width=True)
                    with col2:
                        st.pyplot(create_aspect_ratio_distribution(filtered_detections), use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(create_detection_density_chart(filtered_detections, image_np.shape), use_container_width=True)
                    with col2:
                        st.pyplot(create_roc_curve(filtered_detections), use_container_width=True)
                    
                    st.pyplot(create_bbox_distribution(filtered_detections), use_container_width=True)
                
                with tab8:
                    stats_df = create_class_statistics(filtered_detections)
                    st.subheader("Class-wise Statistics")
                    st.dataframe(stats_df, use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("All Detections")
                    detections_df = pd.DataFrame(filtered_detections)
                    st.dataframe(detections_df, use_container_width=True)
                
                st.markdown("---")
                st.subheader("💾 Export & Download")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    img_buffer = io.BytesIO()
                    annotated_img.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    st.download_button(
                        "📸 Annotated Image",
                        data=img_buffer,
                        file_name="detection_result.png",
                        mime="image/png"
                    )
                
                with col2:
                    if filtered_detections:
                        csv_data = pd.DataFrame(filtered_detections).to_csv(index=False)
                        st.download_button(
                            "📊 Detections CSV",
                            data=csv_data,
                            file_name="detections.csv",
                            mime="text/csv"
                        )
                
                with col3:
                    json_data = export_detections_json(filtered_detections, uploaded_file.name)
                    st.download_button(
                        "📄 Detections JSON",
                        data=json_data,
                        file_name="detections.json",
                        mime="application/json"
                    )
                
                st.markdown("---")
                st.subheader("📋 PDF Report")
                
                if st.button("📑 Generate Full PDF Report"):
                    with st.spinner("Generating PDF..."):
                        stats_df = create_class_statistics(filtered_detections)
                        pdf_buffer = generate_pdf_report(image_pil, annotated_img, filtered_detections, image_quality, health_score, stats_df)
                        
                        if pdf_buffer:
                            st.download_button(
                                "⬇️ Download PDF Report",
                                data=pdf_buffer,
                                file_name="crop_detection_report.pdf",
                                mime="application/pdf"
                            )
            
            else:
                st.info("👆 Upload an image to start")