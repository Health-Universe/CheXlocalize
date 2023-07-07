import os
import base64
import streamlit as st
from argparse import ArgumentParser
import cv2
import glob
import json
import numpy as np
import pandas as pd
import pickle
from pycocotools import mask
import shutil
import subprocess
import sys
import zipfile

sys.path.append(os.path.join(os.path.dirname(__file__), "CheXlocalize"))

from compute_pathology_features import get_geometric_features, main
from eval_constants import LOCALIZATION_TASKS

st.markdown("## Compute Pathology Features")
st.divider()

#
#
#
#
#
#
#

st.subheader("Pathology Features")

# Initialize session state
if 'gt_ann_path' not in st.session_state:
    st.session_state['gt_ann_path'] = None
if 'gt_seg_path' not in st.session_state:
    st.session_state['gt_seg_path'] = None
if 'processing_complete' not in st.session_state:
    st.session_state['processing_complete'] = False
if 'zip_filename' not in st.session_state:
    st.session_state['zip_filename'] = None

# File upload
gt_ann_file = st.file_uploader("Upload Ground-Truth Annotations", type='json', help="**Input:** JSON of Annotations (see Example Input #1).\n\n**Output:** Zip of 4 Pathology Feature CSV Files.")
gt_seg_file = st.file_uploader("Upload Ground-Truth Segmentations", type='json', help="**Input:** JSON of Annotations (see Example Input #2).\n\n**Output:** Zip of 4 Pathology Feature CSV Files.")

# Example Input #1
with st.expander("Example Input #1"):
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

# Example Input #2
with st.expander("Example Input #2"):
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
if gt_ann_file is not None and gt_seg_file is not None:
    run_button = st.button("Run", help="Generating Pathology Features")
    if run_button:
        with st.spinner("Running computation..."):
            # Save files
            gt_ann_path = "./gt_annotations_val.json"
            gt_seg_path = "./gt_segmentations_val.json"
            
            # Read file values
            gt_ann_data = gt_ann_file.getvalue()
            gt_seg_data = gt_seg_file.getvalue()
            
            with open(gt_ann_path, "wb") as f:
                f.write(gt_ann_data)
            with open(gt_seg_path, "wb") as f:
                f.write(gt_seg_data)
            
            # Run script via subprocess
            command = f"python compute_pathology_features.py --gt_ann {gt_ann_path} --gt_seg {gt_seg_path}"
            subprocess.run(command, shell=True)
            
            # Store output files
            output_folder = "Features"
            os.makedirs(output_folder, exist_ok=True)
            
            # Move files to folder
            shutil.move("num_instances.csv", os.path.join(output_folder, "num_instances.csv"))
            shutil.move("area_ratio.csv", os.path.join(output_folder, "area_ratio.csv"))
            shutil.move("elongation.csv", os.path.join(output_folder, "elongation.csv"))
            shutil.move("rec_area_ratio.csv", os.path.join(output_folder, "rec_area_ratio.csv"))
            
            # Create ZIP file
            zip_filename = f"{output_folder}.zip"
            with zipfile.ZipFile(zip_filename, "w") as zipf:
                for root, dirs, files in os.walk(output_folder):
                    for file in files:
                        zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), output_folder))

            # Set session_state values
            st.session_state['gt_ann_path'] = gt_ann_path
            st.session_state['gt_seg_path'] = gt_seg_path
            st.session_state['processing_complete'] = True
            st.session_state['zip_filename'] = zip_filename

            # Cleanup temp files
            os.remove(gt_ann_path)
            os.remove(gt_seg_path)
            shutil.rmtree(output_folder)

# Download button
if st.session_state['processing_complete']:
    st.markdown("---")
    st.markdown("### Download")
    file_bytes = open(st.session_state['zip_filename'], "rb").read()
    st.download_button(label="Download Features Zip", data=file_bytes, file_name=st.session_state['zip_filename'])


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
#

#Plot Pathology Features

import os
import base64
import streamlit as st
from argparse import ArgumentParser
import cv2
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from pycocotools import mask
import seaborn as sns
import shutil
import subprocess
import sys
import zipfile

sys.path.append(os.path.join(os.path.dirname(__file__), "CheXlocalize"))

from plot_pathology_features import plot
from eval_constants import LOCALIZATION_TASKS

st.divider()
st.subheader("Plot Pathology Features")

# Initialize session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "plot_paths" not in st.session_state:
    st.session_state.plot_paths = []
if "zip_file_path" not in st.session_state:
    st.session_state.zip_file_path = ""
if "show_plots" not in st.session_state:
    st.session_state.show_plots = False

# File upload
uploaded_files = st.file_uploader("Upload Pathology Features Files", type = 'csv' , accept_multiple_files=True, help = "**Input:** 4 CSVs of Pathology Features (area_ratio, elongation, num_instances, & rec_area_ratio).\n\n**Output:** Zip of 4 PNG Distribution Plots.")

# Check if four files uploaded
if len(uploaded_files) == 4:
    # Save uploaded files to session state
    st.session_state.uploaded_files = uploaded_files

# File Processing
if st.session_state.uploaded_files:
    run_button = st.button("Run", help="Generating Plot Distribution")
    if run_button:
        with st.spinner("Running"):
            # Create temp dir
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)

            # Save uploaded files to the temp dir
            file_paths = []
            for uploaded_file in st.session_state.uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)

            # Move uploaded files
            for file_path in file_paths:
                shutil.move(file_path, os.path.basename(file_path))

            # Run script via subprocess
            cmd = [sys.executable, "plot_pathology_features.py"]
            cmd.extend(["--features_dir", "."])
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            # Display script output
            st.code(stdout.decode())

            # Show plots if not shown
            if not st.session_state.show_plots:
                plot_paths = []
                for feature in ['n_instance', 'area_ratio', 'elongation', 'irrectangularity']:
                    plot_path = f"{feature}_dist.png"
                    plot_paths.append(plot_path)
                    st.image(plot_path)

                # Save plot paths to session state
                st.session_state.plot_paths = plot_paths

                # Zip file of plots
                zip_file_path = "distribution_plots.zip"
                with zipfile.ZipFile(zip_file_path, "w") as zip_file:
                    for plot_path in plot_paths:
                        zip_file.write(plot_path)

                # Save zip file path to session state
                st.session_state.zip_file_path = zip_file_path

                # Show_plots flag
                st.session_state.show_plots = True

            # Clean up the tem dir
            shutil.rmtree(temp_dir)

# Show plots
if st.session_state.show_plots:
    for plot_path in st.session_state.plot_paths:
        st.image(plot_path)

    # Provide download button for zip
    def download_plots():
        with open(st.session_state.zip_file_path, "rb") as f:
            bytes_data = f.read()
        st.markdown("---")
        st.markdown("### Download")
        st.download_button(label="Download Distribution Plots", data=bytes_data, file_name="plots.zip")

    download_plots()

