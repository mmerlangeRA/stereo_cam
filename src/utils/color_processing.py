import cv2
import numpy as np
from sklearn.cluster import KMeans
from src.road_detection.common import AttentionWindow


red = [28,18,195]
black = [0,0,0]
white = [255,255,255]

def set_color_to_pure_color(color) -> np.ndarray:
    color = np.array(color)

    if color[0]/color[1]>5 and color[0]/color[2]>5:
        color = red
    elif color.min() > 200:
        color = white
    return color

def get_red_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    Returns a mask where pixels are set to 1 if the red component is significantly greater than 
    the green and blue components (specifically, where R/G > 5 and R/B > 5).

    Parameters:
    - image_rgb: Input image in RGB format as a NumPy array of shape (H, W, 3).

    Returns:
    - mask: A binary mask as a NumPy array of shape (H, W), with values 1 or 0.
    """
    # Ensure the image is in float format to avoid integer division issues
    image_rgb = image_rgb.astype(np.float32)

    # Split the image into its red, green, and blue components
    B = image_rgb[:, :, 0]
    G = image_rgb[:, :, 1]
    R = image_rgb[:, :, 2]

    # Avoid division by zero by adding a small epsilon where G or B is zero
    epsilon = 1e-6
    G_safe = np.where(G == 0, epsilon, G)
    B_safe = np.where(B == 0, epsilon, B)

    # Compute the ratios
    R_over_G = R / G_safe
    R_over_B = R / B_safe

    threshold = 1.2

    # Create the mask where both conditions are satisfied
    mask = np.logical_and(R_over_G > threshold, R_over_B > threshold)

    # Convert the boolean mask to an integer mask (1 and 0)
    mask = mask.astype(np.uint8)

    return mask

def get_white_mask(image_rgb: np.ndarray, color_threshold: int = 25, intensity_threshold: int = 80) -> np.ndarray:
    """
    Returns a mask where pixels are set to 1 if the red, green, and blue components are
    approximately the same (within a specified color threshold) and have significant intensity
    (above the intensity threshold).
    
    Parameters:
    - image_rgb: Input image in RGB format as a NumPy array of shape (H, W, 3).
    - color_threshold: Maximum allowed difference between R, G, B components.
    - intensity_threshold: Minimum intensity value for R, G, B components.
    
    Returns:
    - mask: A binary mask as a NumPy array of shape (H, W), with values 1 or 0.
    """
    # Ensure the image is in uint8 format
    image_rgb = image_rgb.astype(np.uint8)
    
    # Split the image into its red, green, and blue components
    B = image_rgb[:, :, 0]
    G = image_rgb[:, :, 1]
    R = image_rgb[:, :, 2]
    
    # Compute the absolute differences between the color channels
    epsilon = 1e-6
    R_safe = np.where(R == 0, epsilon, R)
    G_safe = np.where(G == 0, epsilon, G)
    B_safe = np.where(B == 0, epsilon, B)

    # Compute the ratios
    R_over_G = R_safe / G_safe
    R_over_B = R_safe / B_safe

    R_over_G= np.where(R_over_G >1., R_over_G, 1./R_over_G) 
    R_over_B= np.where(R_over_B >1., R_over_B, 1./R_over_B) 
    threshold = 1.4
    
    # Create a mask where the differences are below the color threshold
    color_mask = np.logical_and(R_over_G <= threshold, R_over_B <= threshold)
    #color_mask = np.logical_and(color_mask, diff_gb <= color_threshold)
    
    # Create a mask where the intensity is above the intensity threshold
    intensity_mask = np.logical_and(R >= intensity_threshold, G >= intensity_threshold)
    intensity_mask = np.logical_and(intensity_mask, B >= intensity_threshold)
    
    # Combine the color and intensity masks
    mask = np.logical_and(color_mask, intensity_mask)
    # Convert the boolean mask to an integer mask (1 and 0)
    mask = mask.astype(np.uint8)
    
    return mask

def get_black_mask(image_rgb: np.ndarray,max_intensity = 30) -> np.ndarray:
   
    # Split the image into its red, green, and blue components
    B = image_rgb[:, :, 0]
    G = image_rgb[:, :, 1]
    R = image_rgb[:, :, 2]


    # Create the mask where both conditions are satisfied
    mask = np.logical_and(B <max_intensity, G <max_intensity)
    mask = np.logical_and(mask, R <max_intensity)
    # Convert the boolean mask to an integer mask (1 and 0)
    mask = mask.astype(np.uint8)

    return mask

def colorize_image_to_pure_colors(image_bgr: np.ndarray, advanced_ops=False) -> np.ndarray:
    h,w = image_bgr.shape[:2]
    # Ensure the image is in uint8 format
    image_bgr = image_bgr.astype(np.uint8)

    # Initialize the output image with an extra alpha channel
    H, W, _ = image_bgr.shape
    output_image = np.zeros((H, W, 4), dtype=np.uint8)  # Shape: (H, W, 4)

    # Get the masks
    mask_red = get_red_mask(image_bgr)
    mask_white = get_white_mask(image_bgr)
    mask_black = get_black_mask(image_bgr)

    if advanced_ops:
        # Apply morphological operations to smooth mask_red
        kernel_size = int(min(h,w)/15.)  # You can adjust this value
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Apply opening to remove small objects (noise)
        #mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        # Apply closing to fill small holes
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #red_contours_mask = np.zeros_like(mask_red)
        if len(contours)>0:
            largest_contour = max(contours, key=cv2.contourArea)
            # Fill the contours on the mask
            cv2.drawContours(mask_red, [largest_contour], -1, color=1, thickness=cv2.FILLED)
        else:
            # Approximate contours to polygons to make edges sharper
            red_contours_mask = np.zeros_like(mask_red)
            approx_contours = []
            for contour in contours:
                epsilon = 0.1 * cv2.arcLength(contour, True)  # Adjust epsilon as needed
                approx = cv2.approxPolyDP(contour, epsilon, True)
                approx_contours.append(approx)

            # Fill the approximated polygons on the mask
            cv2.drawContours(red_contours_mask, approx_contours, -1, color=1, thickness=cv2.FILLED)
            mask_red = red_contours_mask
        
        mask_white = cv2.bitwise_and(mask_white, mask_red)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)>0:
            largest_contour = max(contours, key=cv2.contourArea)
            # Fill the contours on the mask
            cv2.drawContours(mask_white, [largest_contour], -1, color=1, thickness=cv2.FILLED)
        mask_black = cv2.bitwise_and(mask_black, mask_white)

    # Combine the masks
    combined_mask = (mask_red > 0) | (mask_white > 0) | (mask_black > 0)

    # Set the alpha channel: 255 where any mask is true, 0 elsewhere
    alpha_channel = np.where(combined_mask, 255, 0).astype(np.uint8)

    # Set the colors for the masked pixels
    # For red pixels
    output_image[mask_red > 0, :3] = red
    # For white pixels
    output_image[mask_white > 0, :3] = white
    # For black pixels
    output_image[mask_black > 0, :3] = black

    # Set the alpha channel
    output_image[:, :, 3] = alpha_channel

    return output_image

def replace_colors_based_on_main_colors(imageRef_RGB: np.ndarray, image2color_RGB: np.ndarray, debug=True) -> np.ndarray:
    """
    Replaces colors in image2 based on the two main colors found in imageRef.
    
    Parameters:
    - imageRef: The first input image (numpy array in BGR format).
    - image2: The second input image (numpy array in BGR format).
    
    Returns:
    - output_image: The processed image2 with colors replaced based on imageRef's main colors.
    """
    # Step 1: Find the two main colors in imageRef using K-Means clustering
    # Convert imageRef from BGR to RGB color space
    imageRef_rgb = cv2.cvtColor(imageRef_RGB, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to a 2D array of pixels
    pixels = imageRef_rgb.reshape((-1, 3))
    
    # Convert to float type
    pixels = np.float32(pixels)
    
    # Define criteria and apply k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 2  # Number of clusters (main colors)
    attempts = 10
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert centers to uint8 (color values)
    centers = np.uint8(centers)
    
    # The main colors are the cluster centers
    main_colorRef1_rgb = set_color_to_pure_color(centers[0])
    main_colorRef2_rgb = set_color_to_pure_color(centers[1])

    if debug:
        print("rgb",main_colorRef1_rgb,main_colorRef2_rgb)
    
    # Convert main colors to HSV color space for comparison
    main_colorRef1_hsv = cv2.cvtColor(np.uint8([[main_colorRef1_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    main_colorRef2_hsv = cv2.cvtColor(np.uint8([[main_colorRef2_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    
    if debug:
        print("hsv",main_colorRef1_hsv,main_colorRef2_hsv)
        # For debugging: Create images filled with the main colors
        color_patch_size = (100, 100, 3)  # Size of the color patches
        color_patch1 = np.full(color_patch_size, main_colorRef1_rgb, dtype=np.uint8)
        color_patch2 = np.full(color_patch_size, main_colorRef2_rgb, dtype=np.uint8)
        
        # Convert color patches to BGR format for OpenCV display
        color_patch1_bgr = cv2.cvtColor(color_patch1, cv2.COLOR_RGB2BGR)
        color_patch2_bgr = cv2.cvtColor(color_patch2, cv2.COLOR_RGB2BGR)
        
        # Stack the color patches horizontally
        main_colors_image = np.hstack((color_patch1_bgr, color_patch2_bgr))
        
        # Save the main colors image for debugging
        cv2.imshow('main_colors.png', main_colors_image)

    
    # Create empty output image
    output_image = np.zeros_like(image2color_RGB)
    
    
    mask1 = get_red_mask(image2color_RGB)
    
    
    mask2 = get_white_mask(image2color_RGB)
    
    # Apply the masks to set pixels to main colors
    # For pixels similar to main_color1
    output_image[mask1 > 0] = red  # Convert RGB to BGR for OpenCV
    # For pixels similar to main_color2
    output_image[mask2 > 0] = white  # Convert RGB to BGR for OpenCV
    print(np.sum(mask1>0),np.sum(mask2>0))
    # Pixels not matching either color remain black (already set in output_image initialization)
    
    return output_image



