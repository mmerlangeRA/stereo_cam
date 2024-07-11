import argparse
import json
from python_server.components.triangulation_equipolar.main import AutoCalibrationRequest, TriangulationRequest, auto_calibrate_equipoloar, triangulate_equipolar_points


def main():
    parser = argparse.ArgumentParser(description="Process some images.")
    subparsers = parser.add_subparsers(dest="command")

    # Subparser for auto_calibrate
    calibrate_parser = subparsers.add_parser('auto_calibrate')
    calibrate_parser.add_argument('--imgLeft_name', type=str, required=True, help='Name of the left image')
    calibrate_parser.add_argument('--imgRight_name', type=str, required=True, help='Name of the right image')
    calibrate_parser.add_argument('--initial_params', type=json.loads, required=True, help='Initial parameters as a JSON list')
    calibrate_parser.add_argument('--bnds', type=json.loads, required=True, help='Bounds as a JSON list of tuples')
    calibrate_parser.add_argument('--inlier_threshold', type=float, required=False, help='Inlier threshold')
    calibrate_parser.add_argument('--verbose', type=bool, required=False, help='verbose mode')

    # Subparser for triangulatePoints
    triangulate_parser = subparsers.add_parser('triangulatePoints')
    triangulate_parser.add_argument('--keypoints_cam1', type=json.loads, required=True, help='Keypoints from camera 1 as a JSON tuple')
    triangulate_parser.add_argument('--keypoints_cam2', type=json.loads, required=True, help='Keypoints from camera 2 as a JSON tuple')
    triangulate_parser.add_argument('--image_width', type=int, required=True, help='Width of the image')
    triangulate_parser.add_argument('--image_height', type=int, required=True, help='Height of the image')
    triangulate_parser.add_argument('--R', type=json.loads, required=True, help='Rotation matrix parameters x,y,z as a JSON list')
    triangulate_parser.add_argument('--t', type=json.loads, required=True, help='Translation vector x,y,z as a JSON list')
    triangulate_parser.add_argument('--verbose', type=bool, required=False, help='verbose mode')

    args = parser.parse_args()

    if args.command == "auto_calibrate":
        request = AutoCalibrationRequest(
            imgLeft_name=args.imgLeft_name,
            imgRight_name=args.imgRight_name,
            initial_params=args.initial_params,
            bnds=args.bnds,
            inlier_threshold=args.inlier_threshold
        )
        result = auto_calibrate_equipoloar(request)
        print(f"Optimized Parameters: {result}")

    elif args.command == "triangulatePoints":
        request = TriangulationRequest(
            keypoints_cam1=tuple(args.keypoints_cam1),
            keypoints_cam2=tuple(args.keypoints_cam2),
            image_width=args.image_width,
            image_height=args.image_height,
            R=args.R,
            t=args.t,
            verbose=args.verbose
        )
        point1, point2, residual = triangulate_equipolar_points(request)
        print(f"3D Point Camera 1: {point1}")
        print(f"3D Point Camera 2: {point2}")
        print(f"Residual: {residual}")


if __name__ == "__main__":
    main()