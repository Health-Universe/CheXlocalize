import os
import streamlit as st
import pandas as pd
from pathlib import Path
import subprocess
import shutil
import base64
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "CheXlocalize"))

from eval_constants import LOCALIZATION_TASKS
from heatmap_to_segmentation import pkl_to_mask
from precision_recall_specificity import get_results, calculate_precision_recall_specificity, main

st.markdown("## Precision Recall Sensitivity")
st.divider()

# Initialize session state
if 'temp_dir' not in st.session_state:
    st.session_state['temp_dir'] = None

def run_evaluation(gt_path, pred_seg_path, hb_seg_path):
    # Create a temp dir
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    # Save uploaded files to temp dir
    gt_file = temp_dir / "gt_segmentations.json"
    pred_seg_file = temp_dir / "pred_segmentations.json"
    hb_seg_file = temp_dir / "hb_segmentations.json"
    with open(gt_file, "wb") as f:
        f.write(gt_path.read())
    with open(pred_seg_file, "wb") as f:
        f.write(pred_seg_path.read())
    with open(hb_seg_file, "wb") as f:
        f.write(hb_seg_path.read())

    # Cmmd to run evaluation script
    cmd = f"python precision_recall_specificity.py --gt_path {gt_file} --pred_seg_path {pred_seg_file} --hb_seg_path {hb_seg_file} --save_dir {temp_dir}"

    # Run via subprocess
    subprocess.run(cmd, shell=True)
    return temp_dir

# Download
def download_file(file_path, file_name):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">Download CSV</a>'
    return href

# Upload files
gt_path = st.file_uploader("Upload Ground-Truth Segmentations", type="json", help="**Input:** JSON of segmentations (see Example Input #1).\n\n**Output:** CSV of precision, recall/sensitivity, and specificity values.")
pred_seg_path = st.file_uploader("Upload Saliency Method Segmentations", type="json", help="**Input:** JSON of Saliency Method Segmentations.\n\n**Output:** CSV of precision, recall/sensitivity, and specificity values.")
hb_seg_path = st.file_uploader("Upload Human Benchmark Segmentations", type="json", help="**Input:** JSON of Human Benchmark Segmentations.\n\n**Output:** CSV of precision, recall/sensitivity, and specificity values.")

# Example Input #1
with st.expander("Example Input #1"):
    st.code("""
    {
        'patient64622_study1_view1_frontal': {
            'Enlarged Cardiomediastinum': {
                'size': [2320, 2828], # (h, w)
                'counts': '`Vej1Y2iU2c0B?F9G7I6J5K6J6J6J6J6J6H8G9G9J6L4L4L4L4L3M3M3M3L4L4...'
            },
            ....
            'Support Devices': {
                'size': [2320, 2828], # (h, w)
                'counts': 'Xid[1R1ZW29G8H9G9H9F:G9G9G7I7H8I7I6K4L5K4L5K4L4L5K4L5J5L5K...'
            }
        },
        ...
        'patient64652_study1_view1_frontal': {
            ...
        }
    }

    """, language="python")

# File Processing
if gt_path is not None and pred_seg_path is not None and hb_seg_path is not None:
    run_button = st.button("Run", help="Generating Precision Recall Sensitivity Values")
    if run_button:
        with st.spinner("Running"):
            output_dir = run_evaluation(gt_path, pred_seg_path, hb_seg_path)
            output_csv_file = output_dir / "pred_precision_recall_specificity.csv"
            df = pd.read_csv(output_csv_file)

            # Update session state
            st.session_state['temp_dir'] = output_dir

        st.subheader("Evaluation Results")
        st.write(df)

# Check session state for existing temp directory
if st.session_state['temp_dir'] is not None:
    st.markdown("---")
    st.markdown("### Download")
    output_csv_file = st.session_state['temp_dir'] / "pred_precision_recall_specificity.csv"

    # Display the download button

    with open(output_csv_file, 'rb') as file:
            file_data = file.read()
    st.download_button(label="Download Precision Recall Sensitivity Values" , data=file_data, file_name="Precision_Recall_Sensitivity_Values.csv")
