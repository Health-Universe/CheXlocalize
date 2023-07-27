import streamlit as st
import os
import json
import base64
from tqdm import tqdm
from zipfile import ZipFile
import zipfile
import cv2
from argparse import Namespace, ArgumentParser
import base64
import glob
from io import BytesIO
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
import shutil
import subprocess
import sys
import tempfile
import time
import pickle
from pathlib import Path
from pycocotools import mask


sys.path.append(os.path.join(os.path.dirname(__file__), "CheXlocalize"))

from eval import (
    calculate_iou,
    get_ious,
    bootstrap_metric,
    compute_cis,
    create_ci_record,
    get_map,
    get_hitrates,
    get_hb_hitrates,
    evaluate,
)
from eval_constants import LOCALIZATION_TASKS
from utils import CPU_Unpickler

st.markdown("## Evaluate Localization Performance")
st.divider()

#
#
#
#
#

st.subheader("Localization Performance - IOU")

# Initialize session state
if 'gt_seg_path' not in st.session_state:
    st.session_state['gt_seg_path'] = None
if 'pred_path' not in st.session_state:
    st.session_state['pred_path'] = None
if 'processing_complete' not in st.session_state:
    st.session_state['processing_complete'] = False
if 'zip_filename' not in st.session_state:
    st.session_state['zip_filename'] = None

# Metric
st.write("**Metric:** Iou")

# File upload
gt_seg_file = st.file_uploader(
    "Upload Ground-Truth Segmentations File",
    type="json",
    help="**Input:** JSON of segmentations.\n\n**Output:** CSVs of iou/hitmiss results per cxrs, bootstrap results, & summary results.", key="gt_seg_file"
)
pred_file = st.file_uploader(
    "Upload Predicted Segmentations",
    type="json",
    help="**Input:** JSON of Predicted Segmentations.\n\n**Output:** CSVs of iou/hitmiss results per cxrs, bootstrap results, & summary results.", key="pred_file"
)

# Optional
true_pos_only = st.checkbox(
    "True positive only",
    value=True,
    help="If True (default), run evaluation only on the true positive slice of the dataset.", key="1"
)

seed = st.number_input(
    "Random seed",
    value=0,
    help="Random seed to fix for bootstrapping.", key="1a"
)

# File Processing
if gt_seg_file is not None and pred_file is not None:
    run_button = st.button("Run", help="Evaluating IOU Localization Performance", key="b")
    if run_button:
        with st.spinner("Running..."):
            # Save files
            gt_seg_path = "./gt_segmentations_val.json"
            pred_path = "./pred_segmentations_val.json"

            # Read file values
            gt_seg_data = gt_seg_file.getvalue()
            pred_data = pred_file.getvalue()

            with open(gt_seg_path, "wb") as f:
                f.write(gt_seg_data)
            with open(pred_path, "wb") as f:
                f.write(pred_data)

            metric = "iou"
            if_human_benchmark = False

            # Run script via subprocess
            command = (
                f"python eval.py --metric {metric} "
                f"--gt_path {gt_seg_path} "
                f"--pred_path {pred_path} "
                f"--true_pos_only {str(true_pos_only)} "
                f"--if_human_benchmark {str(if_human_benchmark)} "
                f"--seed {str(seed)}"
            )

            subprocess.run(command, shell=True)

            # Store output files
            output_folder = "Localization_Performance"
            os.makedirs(output_folder, exist_ok=True)

            # Move files to folder
            shutil.move("iou_results_per_cxr.csv", os.path.join(output_folder, "iou_results_per_cxr.csv"))
            shutil.move("iou_bootstrap_results_per_cxr.csv", os.path.join(output_folder, "iou_bootstrap_results_per_cxr.csv"))
            shutil.move("iou_summary_results.csv", os.path.join(output_folder, "iou_summary_results.csv"))

            # Create ZIP file
            zip_filename = f"{output_folder}.zip"
            shutil.make_archive(output_folder, "zip", output_folder)

            # Set session_state values
            st.session_state['gt_seg_path'] = gt_seg_path
            st.session_state['pred_path'] = gt_seg_path
            st.session_state['processing_complete'] = True
            st.session_state['zip_filename'] = zip_filename

            # Cleanup temp files
            os.remove(gt_seg_path)
            os.remove(pred_path)
            shutil.rmtree(output_folder)

# Download button
if st.session_state['processing_complete']:
    st.markdown("---")
    st.markdown("### Download")
    file_bytes = open(st.session_state['zip_filename'], "rb").read()
    st.download_button(label="Download Localization Performance Zip", data=file_bytes, file_name=st.session_state['zip_filename'], key="c")



#
#
#
#
#
#
st.divider()
st.subheader("Localization Performance - Hitmiss")

# Initialize session state
if 'gt_seg_path' not in st.session_state:
    st.session_state['gt_seg_path'] = None
if 'pred_path' not in st.session_state:
    st.session_state['pred_path'] = None
if 'processing_complete' not in st.session_state:
    st.session_state['processing_complete'] = False
if 'zip_filename' not in st.session_state:
    st.session_state['zip_filename'] = None

# Metric
st.write("**Metric:** Hitmiss")

# File upload
gt_seg_file = st.file_uploader(
    "Upload Ground-Truth Segmentations File",
    type="json",
    help="**Input:** JSON of segmentations.\n\n**Output:** CSVs of iou/hitmiss results per cxrs, bootstrap results, & summary results.", key="hitmiss"
)
pred_files = st.file_uploader(
    "Upload Directory of Pickle Files",
    type=("pkl",),
    accept_multiple_files=True,
    help="**Input:** Directory containing pickle files of heatmaps.\n\n**Output:** CSVs of iou/hitmiss results per cxrs, bootstrap results, & summary results.", key="hit"
)

# Optional
true_pos_only = st.checkbox(
    "True positive only",
    value=True,
    help="If True (default), run evaluation only on the true positive slice of the dataset.", key="2"
)

seed = st.number_input(
    "Random seed",
    value=0,
    help="Random seed to fix for bootstrapping.", key="2a"
)

# File Processing
if gt_seg_file is not None and pred_files:
    run_button = st.button("Run", help="Evaluating Hitmiss Localization Performance", key="2b")
    if run_button:
        with st.spinner("Running..."):
            try:
                # Save files
                temp_dir = tempfile.mkdtemp()
                st.session_state.temp_dir = temp_dir

                gt_seg_path = os.path.join(temp_dir, gt_seg_file.name)
                with open(gt_seg_path, "wb") as f:
                    f.write(gt_seg_file.getbuffer())

                for pred_file in pred_files:
                    pred_path = os.path.join(temp_dir, pred_file.name)
                    with open(pred_path, "wb") as f:
                        f.write(pred_file.getbuffer())

                    # Read the contents of the pickle file
                    with open(pred_path, "rb") as f:
                        data = pickle.load(f)

                metric = "hitmiss"
                if_human_benchmark = False

                # Run script via subprocess
                command = (
                    f"python eval.py --metric {metric} "
                    f"--gt_path {gt_seg_path} "
                    f"--pred_path {temp_dir} "
                    f"--true_pos_only {str(true_pos_only)} "
                    f"--if_human_benchmark {str(if_human_benchmark)} "
                    f"--seed {str(seed)}"
                )

                subprocess.run(command, shell=True)

                # Store output files
                output_folder = "Localization_Performance"
                os.makedirs(output_folder, exist_ok=True)



                shutil.move("hitmiss_results_per_cxr.csv", os.path.join(output_folder, "hitmiss_results_per_cxr.csv"))
                shutil.move("hitmiss_bootstrap_results_per_cxr.csv", os.path.join(output_folder, "hitmiss_bootstrap_results_per_cxr.csv"))
                shutil.move("hitmiss_summary_results.csv", os.path.join(output_folder, "hitmiss_summary_results.csv"))

                # Create ZIP file
                zip_filename = f"{output_folder}.zip"
                shutil.make_archive(output_folder, "zip", output_folder)

                # Set session_state values
                st.session_state['gt_seg_path'] = gt_seg_path
                st.session_state['pred_path'] = gt_seg_path
                st.session_state['processing_complete'] = True
                st.session_state['zip_filename'] = zip_filename

                # Cleanup temp files
                os.remove(gt_seg_path)
                for pred_file in pred_files:
                    os.remove(os.path.join(temp_dir, pred_file.name))
                shutil.rmtree(output_folder)

            except Exception as e:
                st.error(f"Error occurred: {e}")
# Download button
if st.session_state['processing_complete']:
    st.markdown("---")
    st.markdown("### Download")
    file_bytes = open(st.session_state['zip_filename'], "rb").read()
    st.download_button(label="Download Localization Performance Zip", data=file_bytes, file_name=st.session_state['zip_filename'], key="2c")

#
#
#
#
#
#
st.divider()
st.subheader("Human Benchmark Localization Performance - IOU/Hitmiss")

# Initialize session state
if 'gt_seg_path' not in st.session_state:
    st.session_state['gt_seg_path'] = None
if 'pred_path' not in st.session_state:
    st.session_state['pred_path'] = None
if 'processing_complete' not in st.session_state:
    st.session_state['processing_complete'] = False
if 'zip_filename' not in st.session_state:
    st.session_state['zip_filename'] = None

# Metric
metric = st.selectbox("Select metric", ["iou", "hitmiss"])

# Human Benchmark
st.write("**Human Benchmark:** True")

# File upload
gt_seg_file = st.file_uploader(
    "Upload Ground-Truth Segmentations File",
    type="json",
    help="**Input:** JSON of segmentations.\n\n**Output:** CSVs of iou/hitmiss results per cxrs, bootstrap results, & summary results.", key="whole"
)
pred_file = st.file_uploader(
    "Upload Predicted Segmentations",
    type="json",
    help="**Input:** For Iou, JSON of Predicted Segmentations. For hitmiss, JSON of human annotations.\n\n**Output:** CSVs of iou/hitmiss results per cxrs, bootstrap results, & summary results.", key="both"
)

# Optional

true_pos_only = st.checkbox(
    "True positive only",
    value=True,
    help="If True (default), run evaluation only on the true positive slice of the dataset.", key="3"
)

seed = st.number_input(
    "Random seed",
    value=0,
    help="Random seed to fix for bootstrapping.", key="3a"
)

# File Processing
if gt_seg_file is not None and pred_file is not None:
    run_button = st.button("Run", help="Evaluating Human Benchmark Localization Performance", key="3b")
    if run_button:
        with st.spinner("Running..."):
            # Save files
            gt_seg_path = "./gt_segmentations_val.json"
            pred_path = "./pred_segmentations_val.json"

            # Read file values
            gt_seg_data = gt_seg_file.getvalue()
            pred_data = pred_file.getvalue()

            with open(gt_seg_path, "wb") as f:
                f.write(gt_seg_data)
            with open(pred_path, "wb") as f:
                f.write(pred_data)

            if_human_benchmark = True

            # Run script via subprocess
            command = (
                f"python eval.py --metric {metric} "
                f"--gt_path {gt_seg_path} "
                f"--pred_path {pred_path} "
                f"--true_pos_only {str(true_pos_only)} "
                f"--if_human_benchmark {str(if_human_benchmark)} "
                f"--seed {str(seed)}"
            )

            subprocess.run(command, shell=True)

            # Store output files
            output_folder = "HumanBenchmark_Localization_Performance"
            os.makedirs(output_folder, exist_ok=True)

            # Move files to folder
            filename = f"{metric}_humanbenchmark_results_per_cxr.csv"
            shutil.move(filename, os.path.join(output_folder, filename))
            
            filename2 = f"{metric}_humanbenchmark_bootstrap_results_per_cxr.csv"
            shutil.move(filename2, os.path.join(output_folder, filename2))

            filename3 = f"{metric}_humanbenchmark_summary_results.csv"
            shutil.move(filename3, os.path.join(output_folder, filename3))

            # Create ZIP file
            zip_filename = f"{output_folder}.zip"
            shutil.make_archive(output_folder, "zip", output_folder)

            # Set session_state values
            st.session_state['gt_seg_path'] = gt_seg_path
            st.session_state['pred_path'] = gt_seg_path
            st.session_state['processing_complete'] = True
            st.session_state['zip_filename'] = zip_filename

            # Cleanup temp files
            os.remove(gt_seg_path)
            os.remove(pred_path)
            shutil.rmtree(output_folder)

# Download button
if st.session_state['processing_complete']:
    st.markdown("---")
    st.markdown("### Download")
    file_bytes = open(st.session_state['zip_filename'], "rb").read()
    st.download_button(label="Download Human Benchmark Localization Performance Zip", data=file_bytes, file_name=st.session_state['zip_filename'], key="3c")



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
st.subheader("Calculate Percentage Decrease")

# File upload
hb_bootstrap_results = st.file_uploader('Upload Human Benchmark Bootstrap Results', type='csv', help="**Input:** CSV of 1000 bootstrap samples of IoU or hit/miss for each pathology.\n\n**Output:** CSV of Percentage Decrease.")

pred_bootstrap_results = st.file_uploader("Upload Bootstrap Results", type='csv', help="**Input:** IoU or hit/miss results for each CXR and each pathology.\n\n**Output:** CSV of Percentage Decrease.")

# Metric selection
metric = st.selectbox("Select metric", ["miou", "hitrate"])

# Process Files
if hb_bootstrap_results is not None and pred_bootstrap_results is not None:
    run_button = st.button("Run", help="Generating Percentage Decrease")
    if run_button:
        with st.spinner("Calculating..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                hb_path = os.path.join(temp_dir, "hb_bootstrap_results.csv")
                pred_path = os.path.join(temp_dir, "pred_bootstrap_results.csv")
                
                # Save file content
                with open(hb_path, "wb") as hb_file:
                    hb_file.write(hb_bootstrap_results.read())
                with open(pred_path, "wb") as pred_file:
                    pred_file.write(pred_bootstrap_results.read())

                # Run script via subprocess
                cmd = [
                    sys.executable,
                    "calculate_percentage_decrease.py",
                    "--metric", metric,
                    "--hb_bootstrap_results", hb_path,
                    "--pred_bootstrap_results", pred_path,
                    "--save_dir", temp_dir,
                    "--seed", "0"
                ]
                subprocess.check_call(cmd)

                # Download Button
                result_file = f"{metric}_pct_decrease.csv"
                result_path = os.path.join(temp_dir, result_file)
                result_data = open(result_path, "rb").read()

                # Store result in st.session_state
                st.session_state.result_data = result_data
                st.session_state.result_file = result_file

                # Delete temp files
                shutil.rmtree(temp_dir, ignore_errors=True)

# Display the result if it exists in st.session_state
if "result_data" in st.session_state:
    st.markdown("---")
    st.markdown("### Download")
    st.download_button("Download PCT Decrease", st.session_state.result_data, st.session_state.result_file)
