import streamlit as st
import json
import os
import pandas as pd
import io
import tempfile
from pycocotools import mask
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "CheXlocalize"))

from count_segs import count_segs

st.markdown("## Exploratory Data Analysis")
st.divider()

# File uploader
seg_path = st.file_uploader('Upload', type='json',
                            help="**Input:** JSON of segmentations (see Example Input).\n\n**Output:** CSV of counts.")

# Example Input
with st.expander("Example Input"):
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

# Processing
if seg_path is not None:
    run_button = st.button("Run", help="Generating Segmentation Counts")

    if run_button:
        with st.spinner("Running"):
            seg_dict = json.load(seg_path)

            # Save seg_dict in session state
            if "seg_dict" not in st.session_state:
                st.session_state["seg_dict"] = seg_dict

            # Call count_segs if seg_dict exists
            if "seg_dict" in st.session_state:
                save_dir = tempfile.mkdtemp()

                seg_file_path = os.path.join(save_dir, "seg_file.json")
                with open(seg_file_path, "w") as f:
                    json.dump(st.session_state["seg_dict"], f)

                count_segs(seg_file_path, save_dir)

                output_csv_path = os.path.join(save_dir, "n_segs.csv")
                df = pd.read_csv(output_csv_path)

                # Save df in session state
                st.session_state["df"] = df

                # Clean up temp dir
                os.remove(seg_file_path)
                os.remove(output_csv_path)
                os.rmdir(save_dir)

        if "df" in st.session_state:
            st.dataframe(st.session_state["df"])

# Output CSV data
output = io.BytesIO()
if "df" in st.session_state:
    st.session_state["df"].to_csv(output, index=False)
    output.seek(0)
    
    st.markdown("---")
    st.markdown("### Download")

    # Download CSV
    st.download_button(
        label="Download Segmentation Counts",
        data=output,
        file_name="n_segs.csv",
        mime="text/csv"
    )
