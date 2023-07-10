import streamlit as st
import base64
import tempfile
import subprocess
import numpy as np
import pandas as pd
import json
import os
import sys
import io
import torch
import torch.nn.functional as F
from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw
from pycocotools import mask
import time
import shutil

st.markdown("## Generate Segmnetations")
st.divider()

st.subheader("Saliency Heatmap Segmentations")

sys.path.append(os.path.join(os.path.dirname(__file__), "CheXlocalize"))

from eval_constants import CHEXPERT_TASKS, LOCALIZATION_TASKS
from utils import CPU_Unpickler, encode_segmentation, parse_pkl_filename
from heatmap_to_segmentation import cam_to_segmentation, pkl_to_mask, heatmap_to_mask

# File Upload Required
uploaded_files = st.file_uploader('Upload Heatmaps', type='pkl', accept_multiple_files=True, help="**Input:** Pickle files containing heatmaps (see Example Input).\n\n**Output:** JSON of binary segmentations.")

st.write("")
st.write("Optional Parameters")

# Optional File Uploads
threshold_file = st.file_uploader("Upload Thresholds", type=["csv"], help="**Input:** CSV of Thresholds.\n\n**Output:** JSON of binary segmentations.")
probability_threshold_file = st.file_uploader("Upload Probability Thresholds", type=["csv"], help="**Input:** CSV of Probability Thresholds.\n\n**Output:** JSON of binary segmentations.")

# Optional Parameters
if_smoothing = st.checkbox("Apply Smoothing", help="Check to smooth the pixelated heatmaps via box filtering")
kernel_size = st.number_input("Kernel Size", min_value=0, max_value=10, value=0, help="Kernel size used for box filtering. Make sure to check Smoothing, otherwise no smoothing will be performed.")

# Example Input
with st.expander("Example Input"):
    st.code("""
{
# DenseNet121 + Grad-CAM heatmap <torch.Tensor> of shape (1, 1, h, w)
'map': tensor([[[[1.4711e-06, 1.4711e-06, 1.4711e-06,  ..., 5.7636e-06, 5.7636e-06, 5.7636e-06],
                [1.4711e-06, 1.4711e-06, 1.4711e-06,  ..., 5.7636e-06, 5.7636e-06, 5.7636e-06],
                ...,
             [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 7.9709e-05, 7.9709e-05, 7.9709e-05],
                [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 7.9709e-05, 7.9709e-05, 7.9709e-05]]]]),

# model probability (float)
'prob': 0.02029409697279334,

# one of the ten possible pathologies (string)
'task': Consolidation,

# 0 if ground-truth label for 'task' is negative, 1 if positive (int)
'gt': 0,

# original cxr image
'cxr_img': tensor([[[0.7490, 0.7412, 0.7490,  ..., 0.8196, 0.8196, 0.8118],
              [0.6627, 0.6627, 0.6706,  ..., 0.7373, 0.7137, 0.6941],
                  [0.5137, 0.5176, 0.5294,  ..., 0.6000, 0.5686, 0.5255],
                  ...,
                  [0.7294, 0.7725, 0.7804,  ..., 0.2941, 0.2549, 0.2078],
                  [0.7804, 0.8157, 0.8157,  ..., 0.3216, 0.2824, 0.2510],
                  [0.8353, 0.8431, 0.8549,  ..., 0.3725, 0.3412, 0.3137]],
                  ...
                [[0.7490, 0.7412, 0.7490,  ..., 0.8196, 0.8196, 0.8118],
                  [0.6627, 0.6627, 0.6706,  ..., 0.7373, 0.7137, 0.6941],
                  [0.5137, 0.5176, 0.5294,  ..., 0.6000, 0.5686, 0.5255],
                  ...,
                  [0.7294, 0.7725, 0.7804,  ..., 0.2941, 0.2549, 0.2078],
                  [0.7804, 0.8157, 0.8157,  ..., 0.3216, 0.2824, 0.2510],
                  [0.8353, 0.8431, 0.8549,  ..., 0.3725, 0.3412, 0.3137]]]),

# dimensions of original cxr (w, h)
'cxr_dims': (2022, 1751)
}

    """, language="python")

# Output Path
output_path = os.path.join(tempfile.gettempdir(), "saliency_segmentations.json")

# Check if the session state already exists
if 'download_clicked' not in st.session_state:
    st.session_state.download_clicked = False

# Check if the session state already exists for the temporary directory
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None

# Run Button
if uploaded_files:
    run_button = st.button("Run", help="Generating Segmentations")
    if run_button:
        with st.spinner("Running"):
            if uploaded_files:
                st.spinner("Running")

                # Cleanup the previous temporary directory if it exists
                if st.session_state.temp_dir:
                    shutil.rmtree(st.session_state.temp_dir)
                
                # Create a new temporary directory
                temp_dir = tempfile.mkdtemp()
                st.session_state.temp_dir = temp_dir
                
                # Save uploaded files to temporary directory
                for file in uploaded_files:
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                
                # Build the command for subprocess
                command = ["python", "heatmap_to_segmentation.py", f"--map_dir={temp_dir}", f"--output_path={output_path}"]
                
                if threshold_file:
                    threshold_path = os.path.join(temp_dir, threshold_file.name)
                    with open(threshold_path, "wb") as f:
                        f.write(threshold_file.getbuffer())
                    command.append(f"--threshold_path={threshold_path}")
                
                if probability_threshold_file:
                    prob_threshold_path = os.path.join(temp_dir, probability_threshold_file.name)
                    with open(prob_threshold_path, "wb") as f:
                        f.write(probability_threshold_file.getbuffer())
                    command.append(f"--probability_threshold_path={prob_threshold_path}")
                
                if if_smoothing:
                    command.append("--if_smoothing")
                    command.append(f"--k={kernel_size}")
                
                # Execute the command using subprocess
                subprocess.run(command)
                
                # Set the session state to indicate that the download button has been clicked
                st.session_state.download_clicked = True

# Download Button
if st.session_state.download_clicked:
    st.markdown("---")
    st.markdown("### Download")
    st.download_button("Download Segmentations", data=open(output_path, "rb").read(), file_name="saliency_segmentations.json")

    # Cleanup the temporary directory
    if st.session_state.temp_dir:
        shutil.rmtree(st.session_state.temp_dir)
        st.session_state.temp_dir = None

#
#
#
#
#
#
#
#
#
#
#
#

st.divider()
st.subheader("Human Annotation Segmentations")

sys.path.append(os.path.join(os.path.dirname(__file__), "CheXlocalize"))

from eval_constants import LOCALIZATION_TASKS
from utils import encode_segmentation
from annotation_to_segmentation import create_mask, ann_to_mask

def convert_annotations_to_segmentations(annotations):
    segmentations = {}
    for img_id, img_data in annotations.items():
        if img_id not in segmentations:
            segmentations[img_id] = {}
        for task, polygons in img_data.items():
            img_dims = img_data['img_size']
            try:
                segm_map = create_mask(polygons, img_dims)
                encoded_map = encode_segmentation(segm_map)
                segmentations[img_id][task] = encoded_map
            except TypeError:
                segmentations[img_id][task] = None
    return segmentations

def save_segmentations(segmentations, output_path):
    with open(output_path, "w") as outfile:
        json.dump(segmentations, outfile)
        
def download_json(output_json, output_path):
    with open(output_path, 'w') as f:
        f.write(output_json)
    with open(output_path, 'r') as f:
        data = f.read()
    st.markdown("---")
    st.markdown("### Download")
    st.download_button(
        label="Download Generated Segmentations",
        data=data,
        file_name=output_path,
        mime="application/json"
    )

# File uploader
uploaded_file = st.file_uploader('Upload', type='json', help="**Input:** JSON Human annotations file (see Example Input).\n\n**Output:** JSON of segmentations.")

# Example Input
with st.expander("Example Input"):
    st.code("""
{
    'patient64622_study1_view1_frontal': {
        'img_size': [2320, 2828], # (h, w)
        'Support Devices': [[[1310.68749, 194.47059],
                            [1300.45214, 194.47059],
                            [1290.21691, 201.29412],
                            ...
                            [1310.68749, 191.05883],
                            [1300.45214, 197.88236],
                            [1293.62865, 211.52943]]],
     'Cardiomegaly': [[[1031.58047, 951.35314],
                 [1023.92373, 957.09569],
                 [1012.43856, 964.75249],
              ...
              [1818.31313, 960.92406],
                 [1804.91384, 955.1815],
                 [1789.60024, 951.35314]]],
    ...
    },
    'patient64542_study1_view2_lateral': {
        ...
    }
}
    """, language="python")

# File Processing
if uploaded_file is not None:
    run_button = st.button('Run', key='segmentation_run_button', help="Generating Segmentations")
    
    if run_button:
        with st.spinner('Running...'):
            annotations = json.load(uploaded_file)
            segmentations = convert_annotations_to_segmentations(annotations)
            st.session_state["segmentations"] = segmentations

# Download button
if "segmentations" in st.session_state:
    segmentations = st.session_state["segmentations"]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_output_file:
        temp_output_path = temp_output_file.name
        save_segmentations(segmentations, temp_output_path)

    # Read JSON file
    with open(temp_output_path, 'r') as f:
        output_json = f.read()
    
    # Download button for output
    download_json(output_json, temp_output_path)
