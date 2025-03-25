import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, restoration, filters
from scipy.ndimage import gaussian_gradient_magnitude

def load_image():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return None

def display_image(image, title):
    st.image(image, caption=title, use_column_width=True)

def apply_preprocessing(image, method, params):
    if method == "Histogram Equalization":
        return exposure.equalize_hist(image)
    elif method == "CLAHE":
        return exposure.equalize_adapthist(image, clip_limit=params['clip_limit'], nbins=params['nbins'])
    elif method == "Gamma Correction":
        return exposure.adjust_gamma(image, gamma=params['gamma'])
    elif method == "Log Correction":
        return exposure.adjust_log(image, gain=params['gain'])
    elif method == "Sigmoid Correction":
        return exposure.adjust_sigmoid(image, cutoff=params['cutoff'], gain=params['gain'])
    elif method == "Contrast Stretching":
        p2, p98 = np.percentile(image, (params['low_percentile'], params['high_percentile']))
        return exposure.rescale_intensity(image, in_range=(p2, p98))
    elif method == "Gaussian Denoising":
        return restoration.denoise_gaussian(image, sigma=params['sigma'])
    elif method == "TV Denoising":
        return restoration.denoise_tv_chambolle(image, weight=params['weight'])
    elif method == "Bilateral Filter":
        return restoration.denoise_bilateral(image, sigma_color=params['sigma_color'], sigma_spatial=params['sigma_spatial'])
    elif method == "Gaussian Gradient Magnitude":
        return gaussian_gradient_magnitude(image, sigma=params['sigma'])
    elif method == "Sobel Edge Detection":
        return filters.sobel(image)
    elif method == "Canny Edge Detection":
        return filters.canny(image, sigma=params['sigma'])
    return image

st.title("Image Preprocessing Techniques")

image = load_image()

if image is not None:
    st.sidebar.header("Preprocessing Steps")
    
    preprocessing_steps = []
    
    techniques = {
        "Intensity Transformations": ["Histogram Equalization", "CLAHE", "Gamma Correction", "Log Correction", "Sigmoid Correction", "Contrast Stretching"],
        "Noise Reduction": ["Gaussian Denoising", "TV Denoising", "Bilateral Filter"],
        "Edge Detection": ["Gaussian Gradient Magnitude", "Sobel Edge Detection", "Canny Edge Detection"]
    }
    
    for category, methods in techniques.items():
        with st.sidebar.expander(category):
            for method in methods:
                if st.checkbox(method):
                    preprocessing_steps.append(method)
    
    processed_image = image.copy()
    
    for step in preprocessing_steps:
        st.header(step)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Parameters")
            params = {}
            if step == "CLAHE":
                params['clip_limit'] = st.slider("Clip Limit", 0.01, 1.0, 0.03, 0.01)
                params['nbins'] = st.slider("Number of Bins", 2, 256, 128, 2)
            elif step in ["Gamma Correction", "Log Correction"]:
                params['gamma'] = st.slider("Gamma", 0.1, 5.0, 1.0, 0.1)
            elif step == "Sigmoid Correction":
                params['cutoff'] = st.slider("Cutoff", 0.0, 1.0, 0.5, 0.01)
                params['gain'] = st.slider("Gain", 1, 20, 10)
            elif step == "Contrast Stretching":
                params['low_percentile'] = st.slider("Low Percentile", 0, 100, 2, 1)
                params['high_percentile'] = st.slider("High Percentile", 0, 100, 98, 1)
            elif step in ["Gaussian Denoising", "Gaussian Gradient Magnitude", "Canny Edge Detection"]:
                params['sigma'] = st.slider("Sigma", 0.1, 5.0, 1.0, 0.1)
            elif step == "TV Denoising":
                params['weight'] = st.slider("Weight", 0.01, 1.0, 0.1, 0.01)
            elif step == "Bilateral Filter":
                params['sigma_color'] = st.slider("Sigma Color", 0.1, 1.0, 0.1, 0.1)
                params['sigma_spatial'] = st.slider("Sigma Spatial", 0.1, 1.0, 0.1, 0.1)
        
        with col2:
            st.subheader("Result")
            processed_image = apply_preprocessing(processed_image, step, params)
            display_image(processed_image, step)
    
    st.header("Final Result")
    display_image(processed_image, "Processed Image")
else:
    st.write("Please upload an image to begin preprocessing.")
