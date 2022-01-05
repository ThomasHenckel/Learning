import cv2
import pickle
import numpy as np


with open('camera_calibration.p', 'rb') as f:
    calib_data = pickle.load(f)
    mtx = calib_data["mtx"]
    dist = calib_data["dist"]

img = cv2.imread('camera_cal/calibration1.jpg')
undist = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite( "output_images/calibration1_undistorted.jpg", undist)

img = cv2.imread('test_images/test4.jpg')
undist = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite( "output_images/test4_undistorted.jpg", undist)

img = cv2.imread('test_images/straight_lines1.jpg')
undist = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite( "output_images/straight_lines1.jpg", undist)

imgshape = [720,1280]
# found by plotting out laneline on an image with a straight road
src = np.float32(
    [[588,455],
    [257,682],
    [imgshape[1]-257,682],
    [imgshape[1]-588,455]]) 

sidepadding = 150
dst = np.float32(
    [[sidepadding,0],
    [sidepadding,imgshape[1]],
    [imgshape[0]-sidepadding,imgshape[1]],
    [imgshape[0]-sidepadding,0]])
#use cv2.getPerspectiveTransform() to get M, the transform matrix
M = cv2.getPerspectiveTransform(src,dst)
src = np.array(src, dtype=np.int32)
dst = np.array(dst, dtype=np.int32)

cv2.polylines(undist,[src],True,(0,0,255),thickness=3)

cv2.imwrite( "output_images/straight_lines1_warplines.jpg", undist)

img = cv2.imread('test_images/straight_lines1.jpg')
undist = cv2.undistort(img, mtx, dist, None, mtx)
undist_warped = cv2.warpPerspective(undist,M,(img.shape[0],img.shape[1]))
cv2.polylines(undist_warped,[dst],True,(0,0,255),thickness=3)

cv2.imwrite( "output_images/straight_lines1_warped.jpg", undist_warped)