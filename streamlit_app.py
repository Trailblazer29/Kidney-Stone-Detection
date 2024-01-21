import streamlit as st
from PIL import Image
import numpy as np
from model_inference import KidneyStoneDetectionModel

# Streamlit app content
st.set_page_config(
    page_title="VitalInsight Hub",
    page_icon="../static/logo.jpg",
    layout="wide"
)

# Header
st.header("VitalInsight Hub")
st.image("../static/logo.jpg", width=40, caption="VitalInsight Hub Logo")

# Main content
with st.container():
    st.subheader("Choose Detection Type:")
    detection_type = st.selectbox("", ["Kidney Stone Detection", "Pain Detection"])

    st.subheader("Upload Image:")
    uploaded_image = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

    if st.button("Analyze Image"):
        # Your image processing logic goes here
        st.write("Image analysis result will be displayed here.")

    # Display images if available
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Result section
    result = st.empty()

# Footer
st.markdown("---")
#st.footer("&copy; 2024 VitalInsight Hub. All rights reserved.")
