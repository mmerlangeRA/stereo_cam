from bootstrap import set_paths
set_paths()

import cv2

from src.utils.path_utils import get_output_path
from src.road_detection.common import AttentionWindow
from src.utils.equirectangular.equirectangular_mapper import EquirectangularMapper
from src.utils.equirectangular.SignMatcherClass import SignMatcher


if __name__ == "__main__":
    equirectangular_path_priority = r"C:\Users\mmerl\projects\stereo_cam\data\Photos\test\image_L_307_20240730_143131_519000_7147.jpg"
    signWindow_priority= AttentionWindow(3615,3700,1255,1410,False)
    
    equirectangular_path_stop = r"C:\Users\mmerl\projects\stereo_cam\data\Photos\P1\D_P1_CAM_G_2_EAC.png"
    signWindow_stop= AttentionWindow(3171,3251,1295,1369,False)
    
    if False:
        equirectangular_path = equirectangular_path_priority
        signWindow = signWindow_priority
    else:
        equirectangular_path = equirectangular_path_stop
        signWindow = signWindow_stop
    
    equirect_image =cv2.imread(equirectangular_path)
    equirect_height,equirect_width= equirect_image.shape[:2]
    equirectangularMapper=EquirectangularMapper(equirect_width,equirect_height)
    

    cropped = signWindow.crop_image(equirect_image)
    cv2.imwrite(get_output_path('cropped.png'), cropped)
    #cv2.imshow("cropped",cropped)
    

    signMatcher = SignMatcher()
    sign_id,score = signMatcher.find_matching_sign_id(equirect_image, signWindow, debug=False)
    signImage = signMatcher.get_signImage_from_id(sign_id)

    print(signMatcher.get_image_path_from_id(sign_id))
    # cv2.imshow("sign",cv2.imread(signMatcher.get_image_path_from_id(sign_id)))
    # cv2.waitKey(0)
    #signImage = cv2.imread(r"C:\Users\mmerl\projects\stereo_cam\data\panneaux\AB23-transformed.png")

    estimatedTransform = signMatcher.estimate_initial_sign_transform(
        equirect_image=equirect_image,
        signWindow=signWindow,
        signImage=signImage,
        debug=True
    )

    equirect_sign_image = signMatcher.map_to_equirectangular(
        estimatedTransform, 
        signImage=signImage, 
        equi_width=equirect_width,
         equi_height=equirect_height )
    blended = cv2.addWeighted(equirect_image, 0.5, equirect_sign_image, 0.5, 0)
    cv2.imwrite(get_output_path('equirect_sign_image.png'), blended)


    optimizedTransform = signMatcher.optimize_sign_position_and_orientation(
        equirect_image=equirect_image,
        sign_img=signImage,
        signWindow=signWindow,
        estimatedTransform=estimatedTransform
    )


    print(f"optimizedTransform: x={optimizedTransform}")

    equirect_sign_image = signMatcher.map_to_equirectangular(
        optimizedTransform, 
        signImage=signImage,
        equi_width=equirect_width,
        equi_height=equirect_height)
    blended = cv2.addWeighted(equirect_image, 0.5, equirect_sign_image, 0.5, 0)

    us, vs = signMatcher.get_top_bottom_projected(
        signImage=signImage,
        signTransform=optimizedTransform,
        equi_width=equirect_width,
        equi_height=equirect_height
    )

    for u,v in zip(us,vs):
        cv2.circle(blended, (int(u), int(v)), 5, (0, 0, 255), 1)
        
    cv2.imwrite(get_output_path('optimized.png'), blended)


