import streamlit as st

st.markdown("# CheXLocalize 🩻")
st.divider()

st.markdown("""
    [**Paper**](https://www.nature.com/articles/s42256-022-00536-x) 📖 | 
    [**GitHub**](https://github.com/rajpurkarlab/cheXlocalize) 💻 
    """)
    
st.markdown("""Generate segmentations from saliency method heatmaps or human annotations and evaluate the localization performance of segmentations.""")
             
st.markdown("""
- **Home:** You are here!
- **Exploratory Data Analysis:** Generate the number of CXRs with at least one segmentation.
- **Fine-Tune Thresholds** (optional)**:** Generate segmentation and probability thresholds that maximize mean intersection over union (mIoU) for each pathology.
- **Generate Segmentations:** Generate binary segmentations from saliency heatmaps and human annotations.
- **Evaluate Localization Performance:** Evaluate localization performance mIoU or hit rate and calculate percentage decrease from human benchmark to saliency method.
- **Compute Pathology Features:** Compute pathology (1) number of instances, (2) size, (3) elongation, and (4) irrectangularity.
- **Run Regressions:** Run simple linear regressions on evaluation metrics, pathology features, and the model's probability outputs.
- **Get Precision Recall Sensitivity Values:** Generate precision, recall/sensitivity, and specificity values of the saliency method pipeline and the human benchmark segmentations.
""")

st.divider()
st.markdown("""App Created by [Health Universe](https://www.healthuniverse.com) 🚀
            (Kinal Patel and Mitchell Parker)""")
