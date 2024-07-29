import cv2

def get_cube_sub_images(image:cv2.typing.MatLike)->cv2.typing.MatLike:
    h,w=image.shape[:2]
    sub_w = int(w/4)
    sub_h = int(h/3)
    #sub_images=[image[:sub_h-1,:sub_w-1],image[sub_h:sub_h*2-1,:sub_w-1],image[sub_h:sub_h*2-1,sub_w:sub_w*2-1],image[sub_h:sub_h*2-1,sub_w*2:sub_w*3-1],image[sub_h:sub_h*2-1,sub_w*3:sub_w*4-1]]
    sub_images=[image[sub_h:sub_h*2-1,:sub_w-1],image[sub_h:sub_h*2-1,sub_w:sub_w*2-1],image[sub_h:sub_h*2-1,sub_w*2:sub_w*3-1],image[sub_h:sub_h*2-1,sub_w*3:sub_w*4-1]]
    
    return sub_images

def get_cube_front_image(image:cv2.typing.MatLike)->cv2.typing.MatLike:
    h,w=image.shape[:2]
    sub_w = int(w/4)
    sub_h = int(h/3)
    return image[sub_h:sub_h*2,sub_w*2:sub_w*3]