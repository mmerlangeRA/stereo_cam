import cv2
import numpy as np
import numpy.typing as npt
from typing import Tuple
from sklearn.cluster import KMeans
from src.road_detection.common import AttentionWindow
from src.utils.coordinate_transforms import cartesian_to_spherical_array, spherical_to_equirectangular_array

def cropToWindow(image:cv2.typing.MatLike, window:AttentionWindow)->cv2.typing.MatLike:
    """
    Crops the input image to the given window.

    Parameters:
    - image: Input image (numpy array).
    - window: Tuple (x, y, w, h) defining the window to crop (top-left corner and size).

    Returns:
    - Cropped image.
    """
    x=window.left
    y= window.top
    w=window.right-window.left
    h=window.bottom-window.top
    if y+h>image.shape[0] or x+w>image.shape[1]:
        raise ValueError("Window exceeds image dimensions")
    return image[y:y+h, x:x+w]

def reshapeToWindow(image:cv2.typing.MatLike, window:AttentionWindow, max_width:int)->cv2.typing.MatLike:
    """
    Crops the input image to the given window and resizes it such that 
    the resized width does not exceed max_width, and both width and height
    are multiples of 8.

    Parameters:
    - image: Input image (numpy array).
    - window: Tuple (x, y, w, h) defining the window to crop (top-left corner and size).
    - max_width: Maximum width for resizing the cropped image.

    Returns:
    - Resized image with dimensions (multiple of 8).
    """

    # Step 1: Crop the image to the window
    cropped_image = cropToWindow(image, window=window)
    return cropped_image
    h,w=cropped_image.shape[:2]

    # Step 2: Calculate the resize scale so that the width does not exceed max_width
    resize_scale = min(1, max_width / w)

    # Compute new dimensions
    new_width = int(w * resize_scale)
    new_height = int(h * resize_scale)

    # Ensure the new dimensions are multiples of 8
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8

    # Step 3: Resize the image
    resized_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image

def colorize_disparity_map(disparity:cv2.typing.MatLike)->cv2.typing.MatLike:
    """
    Colorizes the disparity map using a color map.

    Parameters:
    - disparity: Input disparity map (numpy array).

    Returns:
    - Colorized disparity map.
    """
    # Apply color map
    disparity_map_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
            # Apply a colormap (e.g., COLORMAP_JET)
    colorized_disparity_map = cv2.applyColorMap(disparity_map_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            
    return colorized_disparity_map

def is_point_in_blob(binary_image: np.array, point: Tuple[int, int]) -> bool:
    """
    Check if a point is inside the white blob in the binary image.

    Parameters:
    - binary_image: Binary image where the blob is white (255).
    - point: A tuple (x, y) representing the point to check.

    Returns:
    - True if the point is inside the white blob, False otherwise.
    """
    x, y = point
    # Ensure the point is within image bounds
    if 0 <= x < binary_image.shape[1] and 0 <= y < binary_image.shape[0]:
        return binary_image[y, x] == 255  # White pixel means inside the blob
    return False

def move_point_towards_centroid(point: Tuple[int, int], centroid: Tuple[int, int], step: float = 1.0) -> Tuple[int, int]:
    """
    Move a point toward the centroid by a given step.

    Parameters:
    - point: A tuple (x, y) representing the current point.
    - centroid: A tuple (cx, cy) representing the centroid.
    - step: Step size to move toward the centroid.

    Returns:
    - A new point moved toward the centroid.
    """
    x, y = point
    cx, cy = centroid

    # Compute the direction vector from the point to the centroid
    direction = np.array([cx - x, cy - y], dtype=np.float32)
    norm = np.linalg.norm(direction)

    if norm == 0:  # If the point is already at the centroid
        return point

    # Normalize the direction vector and scale by the step size
    direction = (direction / norm) * step

    # Move the point
    new_x = round(x + direction[0])
    new_y = round(y + direction[1])

    return new_x, new_y

def refine_geometrical_shape_in_blob(binary_image: np.array, shape: np.array, centroid: Tuple[int, int]) -> np.array:
    """
    Refine the quadrilateral so that all its points lie inside the white blob in the binary image.

    Parameters:
    - binary_image: Binary image where the blob is white (255).
    - shape: A numpy array of shape (nb_sides, 2) containing the shape points.
    - centroid: The centroid of the shape (or blob) to move points towards if they are outside.

    Returns:
    - A refined shape with points inside the blob.
    """
    refined_quad = shape.copy()

    # Iterate over each point in the quadrilateral
    for i, point in enumerate(refined_quad):
        x, y = point
        # While the point is outside the blob, move it closer to the centroid
        while not is_point_in_blob(binary_image, (x, y)):
            x, y = move_point_towards_centroid((x, y), centroid, step=1.0)
        
        #do once more to be sur
        x, y = move_point_towards_centroid((x, y), centroid, step=2.0)
        # Update the refined quadrilateral with the adjusted point
        refined_quad[i] = [x, y]

    return refined_quad

def find_relevant_corners(contour: np.array, nb_sides: int) -> np.array:
    """
    Finds the nb_sides most relevant corners of a contour. If nb_sides is 0, it detects an ellipse
    and returns the 4 corners of the bounding rectangle.

    Parameters:
    - contour: The input contour (numpy array).
    - nb_sides : The number of sides. If 0, detects an ellipse and returns the 4 corners of the bounding rectangle.

    Returns:
    - A numpy array of shape (nb_sides, 2) or (4, 2) if nb_sides is 0, containing the most relevant corner points.
    """
    if nb_sides == 0:
        print("detecting ellipse   !!")
        # Step 1: Fit an ellipse to the contour
        if len(contour) >= 5:  # Need at least 5 points to fit an ellipse
            ellipse = cv2.fitEllipse(contour)
            (x, y), (MA, ma), angle = ellipse  # Get the ellipse parameters

            # Step 2: Find the bounding rectangle of the ellipse
            bounding_rect = cv2.boundingRect(contour)

            # Step 3: Extract the 4 corners of the bounding rectangle
            x, y, w, h = bounding_rect
            rect_corners = np.array([
                [x, y],           # Top-left
                [x + w, y],       # Top-right
                [x + w, y + h],   # Bottom-right
                [x, y + h]        # Bottom-left
            ], dtype=np.int32)

            return rect_corners

        else:
            raise ValueError("Not enough points to fit an ellipse. Contour must have at least 5 points.")
    
    # Step 1: Get the convex hull of the contour
    hull = cv2.convexHull(contour)
    
    # Step 2: Approximate the convex hull to reduce the number of points
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    
    # If approx already has nb_sides points, return them
    if len(approx) == nb_sides:
        return approx.reshape(nb_sides, 2)
    
    # Step 3: If we have more than nb_sides points, use distance-based filtering
    if len(approx) > nb_sides:
        # Compute the centroid of the contour
        M = cv2.moments(approx)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centroid = np.array([cx, cy])
        
        # Calculate the distance of each point from the centroid
        distances = []
        for point in approx:
            x, y = point[0]
            distance = np.linalg.norm(np.array([x, y]) - centroid)
            distances.append((distance, point))
        
        # Sort the points by distance (descending) and keep the nb_sides farthest
        distances.sort(reverse=True, key=lambda x: x[0])
        most_distant_points = [point[1] for point in distances[:nb_sides]]
        
        # Step 4: Return the nb_sides most distant points as the relevant corners
        return np.array(most_distant_points).reshape(nb_sides, 2)
    
    # If fewer than nb_sides points are returned by approx, fallback to the convex hull's nb_sides farthest points
    return cv2.convexHull(contour, returnPoints=True)[:nb_sides].reshape(nb_sides, 2)

def detect_sign(disparity_map:cv2.typing.MatLike, sign_window:AttentionWindow,nb_sides:int=4)->Tuple[cv2.typing.MatLike,np.array,np.array] :
    """
    Crops the input disparity map image to the specified window, normalizes it, detects the main color,
    and returns a binary black-and-white image corresponding to the cluster with the main color.
    
    Parameters:
    - disparity_map: Disparity map input image (32-bit float numpy array).
    - window: AttentionWindow defining the cropping region.
    - nb_sides: number of sides. 0 means "circle"
    
    Returns:
    - A tuple (binary_image, quadrilateral) where:
        - binary_image: The binary image with the main color blob as white.
        - quadrilateral: A quadrilateral bounding the main color blob in the original image coordinates.
    """
    # Step 1: Crop the image to the window
    cropped_image = cropToWindow(disparity_map, window=sign_window)
    
    # Step 2: Normalize the cropped disparity map to the range [0, 255]
    norm_image = cv2.normalize(cropped_image, None, 0, 255, cv2.NORM_MINMAX)
    
    # Step 3: Convert to 8-bit (required for thresholding)
    norm_image_8bit = np.uint8(norm_image)
    
    # Step 4: Detect the main color (most frequent pixel value in the normalized image)
    # Calculate histogram of pixel intensities
    hist = cv2.calcHist([norm_image_8bit], [0], None, [256], [0, 256])
    
    # Find the most frequent intensity level (main color)
    main_color = np.argmax(hist)
    
    # Step 5: Apply thresholding to isolate the main color blob
    # Use a threshold to create a binary image where the main color is white (255) and others are black (0)
    _, binary_image = cv2.threshold(norm_image_8bit, main_color - 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Return the binary image
    # Step 6: Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 7: Find the largest contour (main blob)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Step 8: Approximate the contour to a polygon with nb_sides points (a quadrilateral)
        shape_points = find_relevant_corners(largest_contour,nb_sides)

        if nb_sides==0 or len(shape_points) == nb_sides:
            # Refine the quadrilateral so that all points are within the blob
            cx, cy =  np.mean(shape_points, axis=0)
            inside_shape = refine_geometrical_shape_in_blob(binary_image, shape_points, (cx, cy))
            disparities=[]
            for p in inside_shape:
                disp=cropped_image[int(p[1]),int(p[0])]
                disparities.append(disp)

            # Add the offsets to the points of the quadrilateral
            x_offset, y_offset = sign_window.left, sign_window.top
            shape_points[:, 0] += x_offset  # Adjust x-coordinates
            shape_points[:, 1] += y_offset
            shape_center = np.mean(shape_points, axis=0)
            
            print(f"shape: {shape_points}")
            
            cx, _ = shape_center  # Extract the x-coordinate of the center

            # Select the two points where the x-coordinate is less than the center's x-coordinate
            left_corners = shape_points[shape_points[:, 0] < cx]
            right_corners = shape_points[shape_points[:, 0] >= cx]
            top_left = left_corners[np.argmin(left_corners[:, 1])]  # Top left corner
            top_right = right_corners[np.argmin(right_corners[:, 1])]  # Top right corner
            bottom_right = right_corners[np.argmax(right_corners[:, 1])]  # Bottom right corner
            bottom_left = left_corners[np.argmax(left_corners[:, 1])]  # Bottom left corner
            
            return binary_image, [top_left,top_right,bottom_right,bottom_left], np.array(disparities)
        else:
            print(f"Could not approximate a shape with {nb_sides}. Returning the binary image only.")
    
    # Return only the binary image if no quadrilateral is found
    return binary_image, None, None

def project_image_to_plane(
    image_2d: npt.NDArray[np.uint8],
    plane_center: npt.NDArray[np.float_],
    plane_normal: npt.NDArray[np.float_],
    plane_up_vector: npt.NDArray[np.float_],
    plane_width: float,
    plane_height: float
) -> npt.NDArray[np.float_]:
    """
    Projects a 2D image onto a plane in 3D space.
    
    Parameters:
    - image_2d: The 2D image as a NumPy array of shape (H, W, C).
    - plane_center: The center of the plane in 3D space (numpy array of shape (3,)).
    - plane_normal: The normal vector of the plane (numpy array of shape (3,)).
    - plane_up_vector: The up vector of the plane (numpy array of shape (3,)).
    - plane_width: The width of the plane in world units.
    - plane_height: The height of the plane in world units.
    
    Returns:
    - points_3d: A NumPy array of shape (H, W, 3) containing 3D points on the plane.
    """
    H, W = image_2d.shape[:2]
    
    # Create local coordinate system for the plane
    # Step 1: Normalize the plane normal vector
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Step 2: Orthogonalize the plane up vector with respect to the plane normal
    plane_up_vector = plane_up_vector - np.dot(plane_up_vector, plane_normal) * plane_normal

    # Step 3: Normalize the adjusted plane up vector
    plane_up_vector = plane_up_vector / np.linalg.norm(plane_up_vector)

    # Step 4: Compute the plane right vector
    plane_right_vector = -np.cross(plane_normal, plane_up_vector)

    # Create grid of local coordinates (s, t)
    s = np.linspace(-plane_width / 2, plane_width / 2, W)
    t = np.linspace(-plane_height / 2, plane_height / 2, H)
    s_grid, t_grid = np.meshgrid(s, t)

    # Compute 3D points on the plane
    points_3d = (plane_center[None, None, :] +
                 s_grid[:, :, None] * plane_right_vector[None, None, :] +
                 t_grid[:, :, None] * plane_up_vector[None, None, :])

    return points_3d

def get_transparency_mask(image: np.ndarray) -> np.ndarray:
    """
    Returns a mask where non-transparent pixels are white (255) and transparent pixels are black (0).
    
    Parameters:
    - image: NumPy array of shape (H, W, 4) for RGBA images.
    
    Returns:
    - mask: Grayscale mask image of shape (H, W).
    """
    # Split the image into color and alpha channels
    if  image.shape[2] == 4:
        b_channel, g_channel, r_channel, alpha_channel = cv2.split(image)
        # Create a mask where alpha channel is greater than 0
        mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)[1]
    elif image.shape[2] == 3:
        mask = np.ones_like(image[:,:,0], dtype=np.uint8) * 255
    else:
        raise ValueError("Image must be RGBA or RGB format.")
    
    return mask

def crop_transparent_borders(image: np.ndarray) -> np.ndarray:
    """
    Crops the transparent borders of an image.

    Parameters:
    - image: NumPy array of shape (H, W, 4) for RGBA images.

    Returns:
    - cropped_image: The image cropped to exclude transparent borders.
    """
    # Check if image has an alpha channel
    if image.shape[2] == 4:
        # Extract the alpha channel
        alpha_channel = image[:, :, 3]
        
        # Create a binary mask where alpha values are greater than 0
        mask = alpha_channel > 0
    elif image.shape[2] == 3:
        mask = np.ones_like(image[:,:,0], dtype=np.uint8) * 255
    else:
        print("The image does not have an alpha channel and is not RGB.")
        return image
        
    # If all pixels are transparent, return the original image
    if not np.any(mask):
        print("The image is fully transparent.")
        return image
    
    # Find coordinates of non-transparent pixels
    coords = np.argwhere(mask)
    
    # Get the bounding box of non-transparent pixels
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # Add 1 because slicing is exclusive at the top
    
    # Crop the image to the bounding box
    cropped_image = image[y0:y1, x0:x1, :]
    return cropped_image
 