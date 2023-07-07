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

st.markdown("## Generate Segmnetations")
st.divider()

st.subheader("Saliency Heatmap Segmentations")
st.write("Coming Soon!")

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
