import sys
import cv2
import os
import numpy as np
import torch
from src.depth_estimation.selective_igev_code.core.igev_stereo import IGEVStereo
from src.depth_estimation.selective_igev_code.core.utils.utils import InputPadder
from src.depth_estimation.depth_estimator import InputPair, StereoMethod, StereoOutput,Config
from src.utils.path_utils import get_pretrained_model_path
# Define the directory to be added
# directory = r'C:\Users\mmerl\projects\stereo_cam\src\depth_estimation\selective_igev_code'
# # Check if the directory is already in the system path
# if directory not in sys.path:
#     sys.path.append(directory)
# directory = r'C:\Users\mmerl\projects\stereo_cam\src\depth_estimation\selective_igev_code\core'
# # Check if the directory is already in the system path
# if directory not in sys.path:
#     sys.path.append(directory)
#print(sys.path)

import torch.nn.functional as F


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Selective_igev(StereoMethod):
    model='igev'
    ckpt='middlebury_finetune.pth'
    def __init__(self, args,config: Config):
        
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        super().__init__("Selective_igev",
                         {},
                         config)
        self.reset_defaults()

        self._loaded_session = None
        self._loaded_model_path = None
        default_model_path = get_pretrained_model_path(self.model,self.ckpt)

        args_dict={
            'hidden_dims':[128]*3,
            'valid_iters':180,
            'n_downsample':2,
            'mixed_precision':False,
            'max_disp':192,
            'n_gru_layers':3,
            'corr_radius':4,
            'corr_levels':2,
            'restore_ckpt':default_model_path
            
        }
        self.args = Args(**args_dict)
        if args is not None:
            for arg in vars(args):
                self.args[arg]=getattr(args, arg)

    def compute_disparity(self, input: InputPair) -> StereoOutput:    
        imgL = torch.from_numpy(input.left_image).permute(2, 0, 1).float()
        imgL=imgL[None].to(self.DEVICE)
        imgR = torch.from_numpy(input.right_image).permute(2, 0, 1).float()
        imgR=imgR[None].to(self.DEVICE)
        model = torch.nn.DataParallel(IGEVStereo(self.args), device_ids=[0])
        model.load_state_dict(torch.load(self.args.restore_ckpt))

        model = model.module
        model.to(self.DEVICE)
        model.eval()
        with torch.no_grad():
            padder = InputPadder(imgL.shape, divis_by=32)
            image1, image2 = padder.pad(imgL, imgR)

            disp = model(image1, image2, iters=self.args.valid_iters, test_mode=True)
            disp = disp.cpu().numpy()
            disp = padder.unpad(disp)
            # print(np.min(disp))
            # print(np.max(disp))
            # print(disp.shape)
            # print(disp.squeeze().shape)
            # file_stem = os.path.basename(imfile1).split('.')[0]
            # filename = os.path.join(output_directory, f'{file_stem}.png')
            # plt.imsave(filename, disp.squeeze(), cmap='jet')
        return StereoOutput(disp.squeeze(), None, None, None)

