import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
import argparse

def calibrate_camera(calibration_images_dir,savePath,nx,ny):
    images = glob.glob(calibration_images_dir)

    # prepare object points
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane

    objp = np.zeros((ny*nx,3),np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) 

    for fname in images:
        img = cv2.imread(fname)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, draw corners
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

    test_img = cv2.imread(images[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, test_img.shape[::-1][1:3],None,None)

    camera_calibration = {'mtx':mtx,'dist':dist}
    with open(savePath, 'wb') as f:
        pickle.dump(camera_calibration, f)

def parse_args():
    """Parse input arguments.
    Parameters
    -------------------
    No parameters.
    Returns
    -------------------
    args: argparser.Namespace class object
        An argparse.Namespace class object contains parameters.
    """
    parser = argparse.ArgumentParser(description='Calibrates a camera using chessboard images')
    parser.add_argument('--i', dest='input',
                        help='Calibration images directory (glob input)',
                        default='camera_cal/calibration*.jpg', type=str)
    parser.add_argument('--o', dest='savePath',
                        help='filename of calibration file',
                        default="camera_calibration.p", type=str)
    parser.add_argument('--nx', dest='nx',
                        help='number of inside corners in x',
                        default=9, type=int)
    parser.add_argument('--ny', dest='ny',
                        help='number of inside corners in y',
                        default=6, type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    print ('Called with args:')
    print (args)

    calibrate_camera(args.input,args.savePath,args.nx,args.ny)