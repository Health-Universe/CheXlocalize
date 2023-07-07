import streamlit as st
from argparse import Namespace
import base64
from io import BytesIO
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from PIL import Image
from pycocotools import mask
import shutil
import sys
import subprocess
import tempfile
import torch.nn.functional as F
from tqdm import tqdm
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "CheXlocalize"))

from eval import compute_cis, create_ci_record
from calculate_percentage_decrease import create_pct_diff_df

st.markdown("## Evaluate Localization Performance")
st.divider()

st.subheader("Localization Performance")
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
