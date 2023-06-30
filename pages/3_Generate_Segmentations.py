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

sys.path.append(os.path.join(os.path.dirname(__file__), "CheXlocalize"))

from eval_constants import CHEXPERT_TASKS, LOCALIZATION_TASKS
from utils import CPU_Unpickler, encode_segmentation, parse_pkl_filename
from heatmap_to_segmentation import cam_to_segmentation, pkl_to_mask, heatmap_to_mask

st.markdown("## Generate Segmentations")
st.divider()

st.subheader("Saliency Heatmap Segmentations")

def process_files(pkl_files):
    segmentations = {}
    for pkl_file in pkl_files:
        pkl_data = pkl_file.read()

        # Create a temp file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(pkl_data)
            temp_file_path = temp_file.name

        try:
            # Default values
            threshold = np.nan
            probability_cutoff = 0
            smoothing = False
            kernel_size = 0

            segmentation = pkl_to_mask(temp_file_path, threshold, probability_cutoff, smoothing, kernel_size)
            encoded_segmentation = encode_segmentation(segmentation)
            pkl_filename = pkl_file.name
            segmentations[pkl_filename] = encoded_segmentation

        except Exception as e:
            print(f"Error occurred during segmentation: {e}")

        # Remove temp file
        Path(temp_file_path).unlink()

    # Convert dictionary to JSON
    segmentations_json = json.dumps(segmentations)

    # Save JSON data to file
    output_file = 'segmentation_data.json'
    output_path = os.path.join(os.getcwd(), output_file)
    with open(output_path, 'w') as f:
        f.write(segmentations_json)

    return output_path


# File uploader
pkl_files = st.file_uploader('Upload', type='pkl', accept_multiple_files=True, help="**Input:** Pickle files containing heatmaps (see Example Input).\n\n**Output:** JSON of binary segmentations.")

# Optional Paths
with st.expander("Optional Paths"):
    st.write ("Coming Soon!")

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
    
# Process Files
run_button = st.button('Run', help="Generating Segmentations")

# Process uploaded files
if pkl_files and run_button:
    with st.spinner("Running"):
        segmentations_output = process_files(pkl_files)
    st.success("Segmentations generated successfully.")

    # Download button for outputted JSON file
    with open(segmentations_output, 'r') as f:
        data = f.read()
    st.download_button(
        label="Download Generated Segmentations",
        data=data,
        file_name="saliency_segmentations.json",
        mime="application/json"
    )

st.divider()

st.subheader("Human Annotation Segmentations")
st.write("Coming Soon!")
