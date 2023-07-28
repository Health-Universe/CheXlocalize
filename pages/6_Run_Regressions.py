import os
import streamlit as st
from argparse import ArgumentParser, Namespace
import pandas as pd
import subprocess
import zipfile
import sys
import shutil
import time

st.markdown("## Run Regressions")
st.divider()

# Options 1
# Dependent Variable - Evaluation metrics.
# Independent Variable - Pathology features.

# Options 2
# Dependent Variable - Evaluation metrics.
# Independent Variable - Model's probability output.

sys.path.append(os.path.join(os.path.dirname(__file__), "CheXlocalize"))

from eval_constants import LOCALIZATION_TASKS
from utils import format_ci, run_linear_regression
from regression_pathology_features import normalize, run_features_regression

st.subheader("Pathology Features Regression")

# Create session state
if "session_state" not in st.session_state:
    st.session_state["session_state"] = {
        "regressions_completed": False,
        "regression_files": [],
    }

def cleanup_temp_files():
    # Remove tempp files
    temp_files = [
        "./pred_iou_results.csv",
        "./pred_hitmiss_results.csv",
        "./hb_iou_results.csv",
        "./hb_hitmiss_results.csv",
        "./regression_results.zip",
    ]
    for file in temp_files:
        if os.path.exists(file):
            os.remove(file)

# File upload
features_files = st.file_uploader(
    "Upload Pathology Features Files",
    type="csv",
    accept_multiple_files=True, help="**Input:** 4 CSVs of Pathology Features (area_ratio, elongation, num_instances, & rec_area_ratio).\n\n**Output:** CSV of Regression Results")

pred_iou_results = st.file_uploader("Upload Saliency Method IoU Results", type="csv", help = "**Input:** CSV of IoU for each CXR & pathology.\n\n**Output:** CSV of Regression Results")
pred_hitmiss_results = st.file_uploader("Upload Saliency Method Hit/Miss Results", help = "**Input:** CSV of hit/miss for each CXR & pathology.\n\n**Output:** CSV of Regression Results")

# Option File Upload & Checkbox
evaluate_hb = st.checkbox("Evaluate Human Benchmark (Optional)", help = "If true, evaluate human benchmark in addition to saliency method.")
if evaluate_hb:
    hb_iou_results = st.file_uploader("Upload Human Benchmark IoU Results", type="csv", help="**Input:** CSV of Human Benchmark IoU results for each CXR & pathology.\n\n**Output:** CSV of Regression Results")
    hb_hitmiss_results = st.file_uploader("Upload Human Benchmark Hit/Miss Results", type="csv", help="**Input:** CSV of Human Benchmark hit/miss results for each CXR & pathology.\n\n**Output:** CSV of Regression Results")

# File Processing
if features_files is not None and pred_iou_results and pred_hitmiss_results is not None:
    run_button = st.button("Run", help="Generating Regression Results")
    if run_button:
        with st.spinner("Running"):
            # Create features directory
            features_dir = "./features"
            os.makedirs(features_dir, exist_ok=True)
            time.sleep(3)

            # Save uploaded feature files
            for file in features_files:
                filename = os.path.join(features_dir, file.name)
                with open(filename, "wb") as f:
                    f.write(file.read())

            # Save other uploaded files
            pred_iou_results_path = "./pred_iou_results.csv"
            pred_hitmiss_results_path = "./pred_hitmiss_results.csv"
            # Save optional uploaded files
            hb_iou_results_path = "./hb_iou_results.csv"
            hb_hitmiss_results_path = "./hb_hitmiss_results.csv"

            if pred_iou_results is not None:
                with open(pred_iou_results_path, "wb") as f:
                    f.write(pred_iou_results.read())

            if pred_hitmiss_results is not None:
                with open(pred_hitmiss_results_path, "wb") as f:
                    f.write(pred_hitmiss_results.read())

            if evaluate_hb:
                if hb_iou_results is not None:
                    with open(hb_iou_results_path, "wb") as f:
                        f.write(hb_iou_results.read())

                if hb_hitmiss_results is not None:
                    with open(hb_hitmiss_results_path, "wb") as f:
                        f.write(hb_hitmiss_results.read())

            # Run regressions
            args = Namespace(
                features_dir=features_dir,
                pred_iou_results=pred_iou_results_path,
                pred_hitmiss_results=pred_hitmiss_results_path,
                evaluate_hb=str(evaluate_hb),
                hb_iou_results=hb_iou_results_path if evaluate_hb else None,
                hb_hitmiss_results=hb_hitmiss_results_path if evaluate_hb else None,
                save_dir=".",
            )
            run_features_regression(args)

            # Set flag to show regressions are done
            st.session_state.session_state["regressions_completed"] = True

# Download regression results
if (
    "regressions_completed" in st.session_state.session_state
    and st.session_state.session_state["regressions_completed"]
):
    regression_files = [
        "regression_features_pred_iou.csv",
        "regression_features_pred_hitmiss.csv",
    ]

    if evaluate_hb:
        regression_files.extend(
            [
                "regression_features_iou_diff.csv",
                "regression_features_hitmiss_diff.csv",
            ]
        )

    zip_path = "./regression_results.zip"

    with zipfile.ZipFile(zip_path, "w") as zipf:
        for file in regression_files:
            zipf.write(file)

    # Download regression results zip file
    with open(zip_path, "rb") as f:
        zip_data = f.read()
    st.markdown("---")
    st.markdown("### Download")
    st.download_button(
        label="Download Regression Results",
        data=zip_data,
        file_name="regression_results.zip",
        mime="application/zip",
    )

    # Cleanup temporary files
    cleanup_temp_files()



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

#Model Assurance
st.divider()
st.subheader("Regression Model Assurance")

st.write("Available upon request")
