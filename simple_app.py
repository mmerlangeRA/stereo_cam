import sys
from matplotlib import pyplot as plt
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from src.road_detection.main import AttentionWindow, get_road_edges_from_eac
import time


st.title("Test détection et dimensionnement")

# Paramètres
uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png"])
cam_height_slider = st.slider("hauteur caméra", 1.5, 2.2, 1.65, 0.01)
max_width_slider = st.slider("max_width", 0, 2048, 2048,128)
limit_left_slider = st.slider("left", 0.0, 1.0, 0.4)
limit_right_slider = st.slider("right", 0.0, 1.0, 0.6)
limit_top_slider = st.slider("top", 0.0, 1.0, 0.3)
limit_bottom_slider = st.slider("bottom", 0.0, 1.0, 0.6)
kernel_slider = st.slider("kernel", 1, 50, 20,1)
degree_slider = st.slider("degree", 1, 3, 2,1)

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Image originale', use_column_width=True)

    # Convert the image to OpenCV format
    image_cv = np.array(image.convert('RGB'))
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    height, width = image_cv.shape[:2]

    max_width = min(max_width_slider,width)
    max_height = int(height/width*max_width)

    img = cv2.resize(image_cv, (max_width, max_height))
    height, width = img.shape[:2]

    # Attention window for segementation and road detection
    limit_left = int(limit_left_slider*width)
    limit_right = int(limit_right_slider*width)
    limit_top = int(limit_top_slider*height)
    limit_bottom = int(limit_bottom_slider*height)

    window = AttentionWindow(limit_left, limit_right, limit_top, limit_bottom)
    window.makeItMultipleOf8()

    st.write("limit_left", window.left, "limit_right", window.right, "limit_top", window.top, "limit_bottom", window.bottom)

    # Processing
    start_time = time.time()
    average_width, first_poly_model, second_poly_model, x, y = get_road_edges_from_eac(img, window, camHeight=cam_height_slider, kernel_width=kernel_slider, degree=degree_slider, debug=True)
    end_time = time.time()
    st.write("Temps calcul (s)",round(end_time-start_time,2))
    st.write("Estimation (m)",round(average_width,2))

    # Debug image

    # Generate y values for plotting the polynomial curves
    y_range = np.linspace(np.min(y), np.max(y), 500)

    # Predict x values using the polynomial models
    x_first_poly = first_poly_model.predict(y_range[:, np.newaxis])
    x_second_poly = second_poly_model.predict(y_range[:, np.newaxis])

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.plot(x_first_poly, y_range, color='red', linewidth=2, label='First Polynomial')
    plt.plot(x_second_poly, y_range, color='blue', linewidth=2, label='Second Polynomial')
    plt.scatter(x, y, color='yellow', s=5, label='Contour Points')
    plt.legend()
    plt.title('Polynomial Curves Fit to Contour Points')
    
    st.pyplot(plt)
    

