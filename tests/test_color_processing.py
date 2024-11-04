from bootstrap import set_paths
set_paths()

import cv2
from python_server.utils.path_helper import get_static_path
from src.road_detection.common import AttentionWindow
from src.utils.color_processing import colorize_image_to_pure_colors, replace_colors_based_on_main_colors

'''
Test to assess more precise position of a road sign
'''


if __name__=="__main__":
    equirectangular_path_priority = r"C:\Users\mmerl\projects\stereo_cam\Photos\test\image_L_307_20240730_143131_519000_7147.jpg"
    signWindow_priority= AttentionWindow(3615,3700,1255,1410,False)
    sign_img_path_priority = r"C:\Users\mmerl\projects\stereo_cam\panneaux\AB23-transformed.png"
    
    equirectangular_path_stop = r"C:\Users\mmerl\projects\stereo_cam\Photos\P1\D_P1_CAM_G_2_EAC.png"
    signWindow_stop= AttentionWindow(3171,3251,1295,1369,False)
    sign_img_path_stop = r"C:\Users\mmerl\projects\stereo_cam\panneaux\France_road_sign_AB4.svg.png"
    
    if False:
        equirectangular_path = equirectangular_path_priority
        signWindow = signWindow_priority
        sign_img_path = sign_img_path_priority
        sign_img = cv2.imread(sign_img_path, cv2.IMREAD_UNCHANGED)
    else:
        equirectangular_path = equirectangular_path_stop
        signWindow = signWindow_stop
        sign_img_path = sign_img_path_stop
        sign_img = cv2.imread(sign_img_path, cv2.IMREAD_UNCHANGED)

    equirect_image =cv2.imread(equirectangular_path)

    cropped = signWindow.crop_image(equirect_image)
    cv2.imwrite(get_static_path("cropped.png"),cropped)

    # Call the function
    processed_cropped = colorize_image_to_pure_colors(cropped,advanced_ops=True)
    processed_sign= colorize_image_to_pure_colors(sign_img, advanced_ops=False)

    # Save or display the processed image
    cv2.imshow('cropped', processed_cropped)
    cv2.imshow('sign_imge', processed_sign)

    processed_cropped = cv2.cvtColor(processed_cropped, cv2.COLOR_RGBA2RGB)
    blended = cv2.addWeighted(cropped, 0.5, processed_cropped, 0.5, 0)
    cv2.imwrite(get_static_path("blended.png"), blended)
    cv2.imshow('cropped', cropped)
    cv2.imshow('processed_cropped', processed_cropped)
    cv2.imshow('blended', blended)

    # Wait for a key press to close the window
    # or to display

    cv2.waitKey(0)
    cv2.destroyAllWindows()
