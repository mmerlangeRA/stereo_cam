import argparse
import json
import os

import cv2
from matplotlib import pyplot as plt
from python_server.components.triangulation_equipolar.main import AutoCalibrationRequest, TriangulationRequest, auto_calibrate_equipoloar, triangulate_equipolar_points
from src.depth_estimation.selective_igev import Selective_igev
from src.depth_estimation.depth_estimator import Calibration, InputPair
from src.depth_estimation.selective_igev_code.demo_imgs import load_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_img', help="path img1", default=None)
    parser.add_argument('-r', '--right_img', help="path to img2", default=None)
    parser.add_argument('--output_directory', help="directory to save output", default=None)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    
    args = parser.parse_args()
    test_igev = Selective_igev(None,None)
    print(args.left_img)
    image1 = cv2.imread(args.left_img)
    image2 = cv2.imread(args.right_img)
    input = InputPair(left_image=image1,right_image=image2,status="started", calibration=None)
    stereo_output = test_igev.compute_disparity(input)
    disp = stereo_output.disparity_pixels
    output_directory= args.output_directory

    if os.path.exists(output_directory) == False:
        os.makedirs(output_directory)

    file_stem = os.path.basename(args.left_img).split('.')[0]
    filename = os.path.join(output_directory, f'{file_stem}.png')
    plt.imsave(filename, disp, cmap='jet')
    calibration_path = r'C:\Users\mmerl\projects\stereo_cam\calibration\stereodemo_calibration.json'
    calibration = Calibration.from_json (open(calibration_path, 'r').read())
    depth_meters = test_igev.depth_meters_from_disparity(disp, calibration)
    filename = os.path.join(output_directory, f'{file_stem}_depth.png')
    plt.imsave(filename, depth_meters, cmap='jet')


if __name__ == "__main__":
    main()