import streamlit as st
import tempfile
import subprocess
import numpy as np
import pandas as pd
import json
import base64
from stqdm import stqdm
import os
from pycocotools import mask
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "CheXlocalize"))

from count_segs import count_segs

st.markdown("## Exploratory Data Analysis")
st.divider()

# File uploader
seg_path = st.file_uploader('Upload', type='json',
                            help="**Input:** JSON of segmentations (see Example Input).\n\n**Output:** CSV of counts.")

with st.expander("Example Input"):
    st.code("""
    {
        'patient64622_study1_view1_frontal': {
            'Enlarged Cardiomediastinum': {
                'size': [2320, 2828], # (h, w)
                'counts': '`Vej1Y2iU2c0B?F9G7I6J5K6J6J6J6J6H8G9G9J6L4L4L4L4L3M3M3M3L4L4...'
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

temp_dir = tempfile.TemporaryDirectory()
save_dir = temp_dir.name

if st.button("Run", help="Generating Segmentation Counts"):
    with st.spinner("Running"):
        if seg_path is not None:
            seg_dict = json.load(seg_path)

            seg_file_path = os.path.join(save_dir, "seg_file.json")
            with open(seg_file_path, "w") as f:
                json.dump(seg_dict, f)

            count_segs(seg_file_path, save_dir)

            output_csv_path = os.path.join(save_dir, "n_segs.csv")
            df = pd.read_csv(output_csv_path)

    st.dataframe(df)

    st.download_button(
        label="Download",
        data=df.to_csv().encode('utf-8'),
        file_name="n_segs.csv",
        mime="text/csv",
        help="Segmentation Counts"
    )

temp_dir.cleanup()
