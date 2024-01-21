import streamlit as st
from PIL import Image
import numpy as np
from model_inference import KidneyStoneDetectionModel

if __name__=="__main__":
    model_path = "./ks_detection.pt"
    original_image = None
    processed_image = None
    analyze_clicked = False

    st.set_page_config(
        page_title="VitalInsight Hub",
        page_icon="logo.jpg"
    )

    st.title("Kidney Stone Detection, by VitalInsight Hub")

    uploaded_file = st.file_uploader(" ", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        byte_data = uploaded_file.getvalue()
        original_image = Image.open(uploaded_file) # PIL image
        processed_image = np.array(original_image.copy()) # RGB image
    
        if st.button("Analyze X-Ray"):
            analyze_clicked = True

    col1, col2 = st.columns([1, 1])

    with col1:
        if original_image:
            st.image(original_image)

    with col2:
        if analyze_clicked:
            model = KidneyStoneDetectionModel(model_path=model_path) # Load model
            model.run_inference(image=original_image) # Run inference
            processed_image = model.annotate_image(image=processed_image) # Annotate image

            st.image(processed_image)
            analyze_clicked = False



    

