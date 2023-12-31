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
import pickle
from PIL import Image, ImageDraw
from pycocotools import mask
import time
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), "CheXlocalize"))

st.markdown("## Fine-Tune Thresholds")
st.divider()

#
#
#
#
#

# Segmentation Thresholds
st.subheader("Segmentation Thresholds")

from eval import calculate_iou
from eval_constants import LOCALIZATION_TASKS
from heatmap_to_segmentation import pkl_to_mask
from tune_heatmap_threshold import compute_miou, tune_threshold

# Pickle File Upload
map_dir = st.file_uploader('Upload Pickle File(s)', type='pkl', accept_multiple_files=True, help="**Input:** Pickle files containing heatmaps (see Example Input #1).\n\n**Output:** CSV of segmentation thresholds.")

# JSON GT File Upload
gt_path = st.file_uploader('Upload Ground-Truth Segmentations File', type='json',
                            help="**Input:** JSON of segmentations (see Example Input #2).\n\n**Output:** CSV of segmentation thresholds.")

# Example Input #1
with st.expander("Example Input #1 - PKL"):
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

# Example Input #2
with st.expander("Example Input #2 - JSON"):
    st.code("""
{
    'patient64622_study1_view1_frontal': {
        'Enlarged Cardiomediastinum': {
        'size': [2320, 2828], # (h, w)
        'counts': '`Vej1Y2iU2c0B?F9G7I6J5K6J6J6J6J6H8G9G9J6L4L4L4L4L3M3M3M3L4L4...'},
        ....
        'Support Devices': {
        'size': [2320, 2828], # (h, w)
        'counts': 'Xid[1R1ZW29G8H9G9H9F:G9G9G7I7H8I7I6K4L5K4L5K4L4L5K4L5J5L5K...'}
    },
    ...
    'patient64652_study1_view1_frontal': {
    ...
    }
}

    """, language="python")

# File processing
if map_dir is not None and gt_path is not None:
    if st.button('Run', help="Generating Segmentation Thresholds"):
        if map_dir is not None and gt_path is not None:
            # Save uploaded files to temp directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded files to session state
                st.session_state['map_dir'] = map_dir
                st.session_state['gt_path'] = gt_path

                map_paths = []
                gt_file_path = os.path.join(temp_dir, 'ground_truth.json')

                # Save GT JSON file
                with open(gt_file_path, 'wb') as f:
                    f.write(gt_path.read())

                # Save heatmap pkl files
                for map_file in map_dir:
                    map_path = os.path.join(temp_dir, map_file.name)
                    with open(map_path, 'wb') as f:
                        f.write(map_file.read())
                    map_paths.append(map_path)

                # Load GT JSON
                with open(gt_file_path, 'r') as f:
                    gt = json.load(f)

                # Generate thresholds and save
                with st.spinner("Running"):
                    tuning_results = pd.DataFrame(columns=['threshold', 'task'])
                    for task in tqdm(sorted(LOCALIZATION_TASKS)):
                        threshold = tune_threshold(task, gt, temp_dir)
                        df = pd.DataFrame([[round(threshold, 1), task]],
                                          columns=['threshold', 'task'])
                        tuning_results = pd.concat([tuning_results, df], axis=0)
                    time.sleep(1)

                # Store tuning_results in session state
                st.session_state['tuning_results'] = tuning_results

# Download button for outputted CSV file
if 'tuning_results' in st.session_state:
    output_csv = st.session_state['tuning_results'].to_csv(index=False)
    output_path = "tuning_results.csv"

    def download_csv():
        with open(output_path, 'w') as f:
            f.write(output_csv)
        with open(output_path, 'r') as f:
            data = f.read()
        st.download_button(
            label="Download Generated Segmentation Thresholds",
            data=data,
            file_name=output_path,
            mime="application/csv"
        )

    st.markdown("---")
    st.markdown("### Download")
    download_csv()

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
#
#
#

# Probability Thresholds
st.divider()
st.subheader("Probability Thresholds")

from eval import calculate_iou
from eval_constants import CHEXPERT_TASKS, LOCALIZATION_TASKS
from heatmap_to_segmentation import cam_to_segmentation
from tune_probability_threshold import compute_miou, find_threshold, main

# PKL Upload
map_files = st.file_uploader('Upload Pickle File(s)', type='pkl', accept_multiple_files=True, help="**Input:** Pickle files containing heatmaps (see Example Input #1).\n\n**Output:** CSV of probability thresholds.")

# JSON Upload
gt_file = st.file_uploader('Upload Ground-Truth Segmentations File', type='json',
    help="**Input:** JSON of segmentations (see Example Input #2).\n\n**Output:** CSV of probability thresholds.")

# Example Input #1
with st.expander("Example Input #1 - PKL"):
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

# Example Input #2
with st.expander("Example Input #2 - JSON"):
    st.code("""
{
    'patient64622_study1_view1_frontal': {
        'Enlarged Cardiomediastinum': {
        'size': [2320, 2828], # (h, w)
        'counts': '`Vej1Y2iU2c0B?F9G7I6J5K6J6J6J6J6H8G9G9J6L4L4L4L4L3M3M3M3L4L4...'},
        ....
        'Support Devices': {
        'size': [2320, 2828], # (h, w)
        'counts': 'Xid[1R1ZW29G8H9G9H9F:G9G9G7I7H8I7I6K4L5K4L5K4L4L5K4L5J5L5K...'}
    },
    ...
    'patient64652_study1_view1_frontal': {
    ...
    }
}

    """, language="python")

# Output Path
output_dir = tempfile.mkdtemp()
threshold_output_path = os.path.join(output_dir, "probability_tuning_results.csv")

# Check if the session state already exists for temp dir
if 'download_clicked' not in st.session_state:
    st.session_state.download_clicked = False
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None
if 'thresholds_path' not in st.session_state:
    st.session_state.thresholds_path = None

if map_files is not None and gt_file is not None:
    run_button = st.button("Run", help="Generating Probability Thresholds")
    if run_button:
        with st.spinner("Running"):
            # Cleanup previous temp dir
            if st.session_state.temp_dir:
                try:
                    shutil.rmtree(st.session_state.temp_dir)
                except FileNotFoundError:
                    pass

            # Create temp dir
            temp_dir = tempfile.mkdtemp()
            st.session_state.temp_dir = temp_dir

            # Save uploaded files to temp dir
            for map_file in map_files:
                map_path = os.path.join(temp_dir, map_file.name)
                with open(map_path, "wb") as f:
                    f.write(map_file.getbuffer())

            gt_path = os.path.join(temp_dir, gt_file.name)
            with open(gt_path, "wb") as f:
                f.write(gt_file.getbuffer())

            # Run command
            command = [
                "python",
                "tune_probability_threshold.py",
                f"--map_dir={temp_dir}",
                f"--gt_path={gt_path}",
                f"--save_dir={output_dir}"
            ]
            subprocess.run(command)

            # Check if the thresholds file is generated
            thresholds_path = os.path.join(output_dir, "probability_tuning_results.csv")
            if os.path.exists(thresholds_path):
                st.session_state.thresholds_path = thresholds_path
                st.session_state.download_clicked = True

# Download Button
if st.session_state.download_clicked:
    st.markdown("---")
    st.markdown("### Download")
    thresholds_data = open(st.session_state.thresholds_path, "rb").read()
    st.download_button(
        "Download Generated Probability Thresholds",
        data=thresholds_data,
        file_name="probability_tuning_results.csv"
    )

