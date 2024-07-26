import sys
import cv2
import numpy as np
import torch
from src.depth_estimation.selective_igev_code.core.igev_stereo import IGEVStereo
from src.depth_estimation.selective_igev_code.core.utils.utils import InputPadder
from src.depth_estimation.depth_estimator import InputPair, StereoMethod, StereoOutput,Config
# Define the directory to be added
# directory = r'C:\Users\mmerl\projects\stereo_cam\src\depth_estimation\selective_igev_code'
# # Check if the directory is already in the system path
# if directory not in sys.path:
#     sys.path.append(directory)
# directory = r'C:\Users\mmerl\projects\stereo_cam\src\depth_estimation\selective_igev_code\core'
# # Check if the directory is already in the system path
# if directory not in sys.path:
#     sys.path.append(directory)
print(sys.path)
import os
import torch.nn.functional as F

DEVICE = 'cuda'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Selective_igev(StereoMethod):
    def __init__(self, args,config: Config):
        super().__init__("Selective_igev",
                         {},
                         config)
        self.reset_defaults()

        self._loaded_session = None
        self._loaded_model_path = None
        args_dict={
            'hidden_dims':[128]*3,
            'valid_iters':180,
            'n_downsample':2,
            'mixed_precision':False,
            'max_disp':192,
            'n_gru_layers':3,
            'corr_radius':4,
            'corr_levels':2,
            'restore_ckpt':r'C:\Users\mmerl\projects\stereo_cam\src\depth_estimation\selective_igev_code\pretrained_models\middlebury_finetune.pth'
            
        }
        self.args = Args(**args_dict)
        if args is not None:
            for arg in vars(args):
                self.args[arg]=getattr(args, arg)



    def compute_disparity(self, input: InputPair) -> StereoOutput:    
        imgL = torch.from_numpy(input.left_image).permute(2, 0, 1).float()
        imgL=imgL[None].to(DEVICE)
        imgR = torch.from_numpy(input.right_image).permute(2, 0, 1).float()
        imgR=imgR[None].to(DEVICE)
        model = torch.nn.DataParallel(IGEVStereo(self.args), device_ids=[0])
        model.load_state_dict(torch.load(self.args.restore_ckpt))

        model = model.module
        model.to(DEVICE)
        model.eval()
        with torch.no_grad():
            padder = InputPadder(imgL.shape, divis_by=32)
            image1, image2 = padder.pad(imgL, imgR)

            disp = model(image1, image2, iters=self.args.valid_iters, test_mode=True)
            disp = disp.cpu().numpy()
            disp = padder.unpad(disp)
            print(np.min(disp))
            print(np.max(disp))
            print(disp.shape)
            print(disp.squeeze().shape)
            # file_stem = os.path.basename(imfile1).split('.')[0]
            # filename = os.path.join(output_directory, f'{file_stem}.png')
            # plt.imsave(filename, disp.squeeze(), cmap='jet')
        return StereoOutput(disp.squeeze(), None, None, None)

