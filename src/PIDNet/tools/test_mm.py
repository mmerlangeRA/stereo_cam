# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------

import glob
import argparse
import cv2
import os
import numpy as np
import _init_paths
import models
import torch
import torch.nn.functional as F
from PIL import Image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

color_map = [(128, 64,128),
             (244, 35,232),
             ( 70, 70, 70),
             (102,102,156),
             (190,153,153),
             (153,153,153),
             (250,170, 30),
             (220,220,  0),
             (107,142, 35),
             (152,251,152),
             ( 70,130,180),
             (220, 20, 60),
             (255,  0,  0),
             (  0,  0,142),
             (  0,  0, 70),
             (  0, 60,100),
             (  0, 80,100),
             (  0,  0,230),
             (119, 11, 32)]


def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image

def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict = False)
    
    return model

def get_image_paths(folder):
    print("folder", folder)
    # Define the image file extensions you want to include
    extensions = ['.png']
    image_paths = []
    for ext in extensions:
        search_path = os.path.join(folder, f'*{ext}')
        print("Search path:", search_path)
        found_paths = glob.glob(search_path, recursive=False)
        print(f"Found {len(found_paths)} files with extension {ext}")
        image_paths.extend(found_paths)
        
    return image_paths

if __name__ == '__main__':
    root = os.getcwd()
    root = os.path.join(root, 'PIDNet')
    image_folder_path = os.path.join(root, 'samples')
    extension =".png"
    print("Image folder: ", image_folder_path)
    images_list =get_image_paths(image_folder_path)
    sv_path = image_folder_path+'outputs/'
    a = 'pidnet-l'
    c= True

    model_path  = os.path.join(root, 'pretrained_models')
    model_path = os.path.join(model_path, 'cityscapes')
    model_path = os.path.join(model_path, 'PIDNet_L_Cityscapes_test.pt')
    output_path = os.path.join(root, 'output')
    
    model = models.pidnet.get_pred_model(a, 19 if c else 11)
    model = load_pretrained(model, model_path).cuda()
    model.eval()
    with torch.no_grad():
        print("images_list",images_list)
        for img_path in images_list:
            img_name = img_path.split("\\")[-1]
            print("Processing: ", img_name)
            # Read image
            img = cv2.imread(os.path.join(image_folder_path, img_name),
                               cv2.IMREAD_COLOR)
            sv_img = np.zeros_like(img).astype(np.uint8)
            img = input_transform(img)
            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).unsqueeze(0).cuda()
            pred = model(img)
            pred = F.interpolate(pred, size=img.size()[-2:], 
                                 mode='bilinear', align_corners=True)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            
            for i, color in enumerate(color_map):
                for j in range(3):
                    sv_img[:,:,j][pred==i] = color_map[i][j]
            sv_img = Image.fromarray(sv_img)
            
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            sv_img.save(os.path.join(output_path, img_name))
            
            
            
        
        