import cv2
from matplotlib import pyplot as plt
import numpy as np
from src.road_detection.main import AttentionWindow, compute_road_width_from_eac


if __name__ == '__main__':
    img_path = r'C:\Users\mmerl\projects\stereo_cam\Photos\P5\D_P5_CAM_G_0_EAC.png'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    height, width = img.shape[:2]

    max_width = 2048
    max_height = int(height/width*max_width)

    img = cv2.resize(img, (max_width, max_height))
    height, width = img.shape[:2]

    # Attention window for segementation and road detection
    window_Width= int(width/3)
    limit_left = window_Width
    limit_right = width-window_Width-int(width/10)
    limit_top = int(height/2.5)
    limit_bottom = height-int(height/2.5)

    window = AttentionWindow(limit_left, limit_right, limit_top, limit_bottom)

    #processing
    average_width,first_poly_model, second_poly_model,x,y = compute_road_width_from_eac(img,window,camHeight=1.65,degree=1,debug=True)

    # Generate y values for plotting the polynomial curves
    y_range = np.linspace(np.min(y), np.max(y), 500)

    # Predict x values using the polynomial models
    x_first_poly = first_poly_model.predict(y_range[:, np.newaxis])
    x_second_poly = second_poly_model.predict(y_range[:, np.newaxis])

    # Plot the polynomial curves on the image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.plot(x_first_poly, y_range, color='red', linewidth=2, label='First Polynomial')
    plt.plot(x_second_poly, y_range, color='blue', linewidth=2, label='Second Polynomial')
    plt.scatter(x, y, color='yellow', s=5, label='Contour Points')
    plt.legend()
    plt.title('Polynomial Curves Fit to Contour Points')
    plt.savefig(r'C:\Users\mmerl\projects\stereo_cam\output\polynomial_fit.png')
    plt.show()
    
    
    print("average width",average_width)
