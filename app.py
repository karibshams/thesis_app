import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import io

# Set page config
st.set_page_config(page_title="Object Detection", layout="wide")
st.title("🌾 Crop Detection System - YOLOv11")
st.markdown("Detect Sunflower and Rice Objects with Advanced Visualizations")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model file upload
    uploaded_model = st.file_uploader(
        "📤 Upload best.pt Model",
        type=["pt"],
        help="Upload your trained YOLOv11 best.pt weights"
    )
    
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
    
    dataset_type = st.radio(
        "Dataset Type",
        ["Sunflower", "Rice", "Both"],
        help="Select which dataset to filter results"
    )
    
    viz_type = st.multiselect(
        "Visualization Types",
        ["Detection Boxes", "Confidence Heatmap", "Class Distribution", "Predictions Table"],
        default=["Detection Boxes", "Predictions Table"]
    )

# Load model from uploaded file
@st.cache_resource
def load_model_from_file(model_file):
    try:
        # Save uploaded file temporarily
        with open("temp_model.pt", "wb") as f:
            f.write(model_file.getbuffer())
        
        model = YOLO("temp_model.pt")
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Detection function
def run_detection(image_np, model, conf, iou):
    results = model(image_np, conf=conf, iou=iou, imgsz=640)
    return results

# Convert PIL to numpy
def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Convert numpy to PIL
def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

# Draw detection boxes
def draw_detections(image_np, results, conf_threshold):
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
            
            # Draw rectangle
            cv2.rectangle(image_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{cls_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image_annotated, (x1, y1 - 25), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image_annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            detections.append({
                "Class": cls_name,
                "Confidence": f"{conf:.4f}",
                "X1": x1,
                "Y1": y1,
                "X2": x2,
                "Y2": y2
            })
    
    return cv2_to_pil(image_annotated), detections

# Confidence heatmap visualization
def create_confidence_heatmap(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    confidences = []
    class_names = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            cls_name = result.names[cls]
            confidences.append(conf)
            class_names.append(cls_name)
    
    if confidences:
        ax.barh(range(len(confidences)), confidences, color='skyblue')
        ax.set_yticks(range(len(confidences)))
        ax.set_yticklabels([f"{c} {i+1}" for i, c in enumerate(class_names)])
        ax.set_xlabel("Confidence Score")
        ax.set_title("Detection Confidence Scores")
        ax.set_xlim([0, 1])
        for i, v in enumerate(confidences):
            ax.text(v + 0.02, i, f"{v:.3f}", va='center')
    
    return fig

# Class distribution pie chart
def create_class_distribution(results):
    class_counts = {}
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            cls_name = result.names[cls]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if class_counts:
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
        ax.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title("Class Distribution")
    
    return fig

# Main app
st.markdown("---")

# Check if model is uploaded
if uploaded_model is None:
    st.warning("⚠️ Please upload a best.pt model file in the sidebar to get started")
else:
    # Load the uploaded model
    model = load_model_from_file(uploaded_model)
    
    if model is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "📤 Upload an Image",
                type=["jpg", "jpeg", "png", "bmp"],
                help="Upload an image for object detection"
            )
        
        with col2:
            st.write("")
            st.write("")
            if st.button("🔍 Run Detection", key="detect_btn", use_container_width=True):
                st.session_state.run_detection = True
        
        # Process image
        if uploaded_file is not None:
            # Read image
            image_pil = Image.open(uploaded_file).convert('RGB')
            image_np = pil_to_cv2(image_pil)
            
            # Run detection
            st.info("🔄 Running detection...")
            results = run_detection(image_np, model, conf_threshold, iou_threshold)
            
            # Get annotated image and detections
            annotated_img, detections = draw_detections(image_np, results, conf_threshold)
            
            # Display results
            st.success(f"✅ Detection Complete! Found {len(detections)} objects")
            
            st.markdown("---")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(
                ["📸 Detection", "📊 Analytics", "📈 Confidence", "📋 Details"]
            )
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Image")
                    st.image(image_pil, use_column_width=True)
                with col2:
                    st.subheader("Detected Objects")
                    st.image(annotated_img, use_column_width=True)
            
            with tab2:
                if len(detections) > 0:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(create_class_distribution(results), use_container_width=True)
                    with col2:
                        st.pyplot(create_confidence_heatmap(results), use_container_width=True)
                else:
                    st.info("No objects detected to display analytics")
            
            with tab3:
                if len(detections) > 0:
                    st.pyplot(create_confidence_heatmap(results), use_container_width=True)
                else:
                    st.info("No detections for confidence visualization")
            
            with tab4:
                if len(detections) > 0:
                    st.subheader("Detection Results")
                    st.dataframe(detections, use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("Summary Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Detections", len(detections))
                    
                    with col2:
                        avg_conf = np.mean([float(d["Confidence"]) for d in detections])
                        st.metric("Average Confidence", f"{avg_conf:.4f}")
                    
                    with col3:
                        max_conf = max([float(d["Confidence"]) for d in detections])
                        st.metric("Max Confidence", f"{max_conf:.4f}")
                else:
                    st.info("No objects detected")
            
            st.markdown("---")
            
            # Download results
            st.subheader("💾 Download Results")
            
            # Convert annotated image to bytes
            img_byte_arr = io.BytesIO()
            annotated_img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            st.download_button(
                label="⬇️ Download Annotated Image",
                data=img_byte_arr,
                file_name="detection_result.png",
                mime="image/png"
            )
            
            # Download detections as CSV
            if len(detections) > 0:
                import pandas as pd
                df = pd.DataFrame(detections)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Detections (CSV)",
                    data=csv,
                    file_name="detections.csv",
                    mime="text/csv"
                )
        
        else:
            st.info("👆 Upload an image to start detection")
            
            # Show sample info
            with st.expander("ℹ️ How to use this app"):
                st.write("""
                1. **Upload Model**: Upload your best.pt model in the sidebar
                2. **Configure Settings**: Adjust confidence and IOU thresholds in the sidebar
                3. **Upload Image**: Click to upload a JPG, PNG, or BMP image
                4. **Run Detection**: The app will automatically run detection on upload
                5. **View Results**: 
                   - See original vs detected image
                   - Analyze confidence scores
                   - View class distribution
                   - Download annotated image and results
                """)