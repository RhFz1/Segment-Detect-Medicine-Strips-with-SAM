import streamlit as st
import time
from PIL import Image
from main import test_inference

def display_medicine_info(result):
    if not result:
        st.warning("No medicines detected.")
        return
    
    for i, (medicine_name, info) in enumerate(result.items()):
        with st.expander(f"{i + 1}. {medicine_name}", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Basic Information:**")
                st.write(f"Count: {info.get('Count', 'N/A')}")
                st.write(f"Area: {info.get('Area', 'N/A'):.2f}")
                st.write(f"Strength: {info.get('Strength', 'N/A')}")
                st.write(f"Price: {info.get('Price', 'N/A')}")
                st.markdown("**Detailed Information:**")
                st.write(f"Formula/Ingredients: {info.get('Formula/Ingredients', 'N/A')}")
                st.write(f"Manufacturer: {info.get('Manufacturer', 'N/A')}")
                st.write(f"Dosage: {info.get('Dosage', 'N/A')}")
            
            with col2:
                st.markdown("**Dates:**")
                st.write(f"Expiry Date: {info.get('Expiry_Date', 'N/A')}")
                st.write(f"Manufacture Date: {info.get('Manufacture_Date', 'N/A')}")
            
            

# Set up the page configuration
st.set_page_config(page_title="Med Strip Detection", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .title-style {
        font-size: 48px;
        font-weight: bold;
        color: #333333;
        text-align: center;
        font-family: 'Poppins', sans-serif;
        margin-bottom: 30px;
        letter-spacing: 1.5px;
    }
    .stFileUploader {
        border: 2px solid #5c636a;
        border-radius: 15px;
        padding: 10px;
        background-color: #ffffff;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
    }
    .preview-image {
        display: block;
        margin-top: 10px;
        border: 2px solid #5c636a;
        border-radius: 10px;
        width: 200px;
        height: auto;
    }
    .info-box {
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #5c636a;
        background: #f0f2f6;
        font-size: 18px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        color: #333333;
        margin-top: 20px;
        transition: all 0.3s ease-in-out;
        text-align: center;
        max-height: 300px;
        overflow-y: auto;
    }
    .info-box:hover {
        background: #e0e3e7;
        color: #000;
        border-color: #5c636a;
        box-shadow: 0px 6px 18px rgba(0, 0, 0, 0.15);
    }
    .spinner {
        width: 50px;
        height: 50px;
        border: 5px solid #f3f3f3;
        border-top: 5px solid #5c636a;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
""", unsafe_allow_html=True)

# Display the title
st.markdown('<div class="title-style">Med Strip Detection</div>', unsafe_allow_html=True)

# Create a section for the file uploader and image preview
st.subheader("Upload your image:")
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

# Preview below the upload section
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Preview", use_column_width=True)  # Preview below uploader

# Extracting information
if uploaded_file is not None:
    t0 = time.time()
    with st.spinner("Extracting information..."):
        result = test_inference(image=image)  # Call to inference function

    t1 = time.time()
    time_taken = t1 - t0

    display_medicine_info(result)  # Display the result
    
    st.write(f"Time taken for inference: {time_taken:.2f} seconds")
else:
    st.info("Upload an image to see the extracted information.")