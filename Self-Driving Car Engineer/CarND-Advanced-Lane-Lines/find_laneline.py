import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import cv2
import glob
import pickle
import argparse

from moviepy.editor import VideoFileClip

class Line():
    def __init__(self):
        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 3/165 # meters per pixel in y dimension (standard lane line in m / the pixel length of one lane line in px plotted out on the transformed image)
        self.xm_per_pix = 3.7/440 # meters per pixel in x dimension (standard line width in m / the with of the lane line in px in the transformed image)
        self.n = 10
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # polynomial coefficients for the last n fits
        self.recent_fit = [] 
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

    def fit(self):
        self.current_fit = np.polyfit(self.ally,self.allx,2)
        if(len(self.recent_fit) >= self.n):
            self.recent_fit.pop(0)
        self.recent_fit.append(self.current_fit)

        self.best_fit = np.mean(self.recent_fit, axis=0)
    
    def solve(self,y):
        return self.current_fit[0]*y**2 + self.current_fit[1]*y + self.current_fit[2]

    def solve_average(self,y):
        return self.best_fit[0]*y**2 + self.best_fit[1]*y + self.best_fit[2]

    def append_line(self,fit_x,y):
        radius_of_current_curvature = measure_curvature(fit_x*self.xm_per_pix,y*self.ym_per_pix)
        if(radius_of_current_curvature > 100): # if raduis is less than 100m it is properly a bad meassurement
            if(len(self.recent_xfitted) >= self.n):
                self.recent_xfitted.pop(0)
            self.recent_xfitted.append(fit_x)

            self.bestx = np.mean(self.recent_xfitted, axis=0)
            self.radius_of_curvature = measure_curvature(self.bestx*self.xm_per_pix,y*self.ym_per_pix)



def hls_gradient_select(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary,color_binary

def load_camera_calibration(calibrationFile):
    # undistort variables
    with open(calibrationFile, 'rb') as f:
        calib_data = pickle.load(f)
        mtx = calib_data["mtx"]
        dist = calib_data["dist"]
    return mtx,dist


def get_perspective_transform_vector(imgshape):
    # Perspective Transform variables
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
    return cv2.getPerspectiveTransform(src,dst)

def calculate_curvature(fit,y):
    A = fit[0]
    B = fit[1]
    return np.power((np.square(2*A*y+B)+1),1.5)/np.abs(2*A)

def measure_curvature(x,y):
    fit_cr = np.polyfit(y, x, 2)
    curverad = calculate_curvature(fit_cr,np.max(y)) 
    return curverad

def measure_distance_to_center(left_line,right_line,imagewidth):
    width_of_road=right_line-left_line
    center_of_road =left_line + width_of_road/2
    offset = center_of_road - imagewidth/2
    return offset

class FindLaneLines():
    def __init__(self,calibrationFile,debug):
        self.M = get_perspective_transform_vector([720,1280])
        self.ploty = None
        self.left_lane_line = Line()
        self.right_lane_line = Line()
        self.lost = True
        self.mtx,self.dist = load_camera_calibration(calibrationFile)
        self.debug = debug

    def find_lane_pixels(self,binary_warped):

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Set the width of the windows +/- margin
        margin = 100
        if self.lost:
            # Take a histogram of the bottom half of the image
            histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0]//2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            # HYPERPARAMETERS
            # Choose the number of sliding windows
            nwindows = 9

            # Set minimum number of pixels found to recenter window
            minpix = 50

            # Set height of windows - based on nwindows above and image shape
            window_height = np.int(binary_warped.shape[0]//nwindows)

            # Current positions to be updated later for each window in nwindows
            leftx_current = leftx_base
            rightx_current = rightx_base

            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window+1)*window_height
                win_y_high = binary_warped.shape[0] - window*window_height
                # Find the four below boundaries of the window
                win_xleft_low = leftx_current-margin
                win_xleft_high = leftx_current+margin
                win_xright_low = rightx_current-margin 
                win_xright_high = rightx_current+margin 
                
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = self.get_good_inds(nonzeroy,nonzerox,win_y_low,win_y_high,win_xleft_low,win_xleft_high)
                good_right_inds =self.get_good_inds(nonzeroy,nonzerox,win_y_low,win_y_high,win_xright_low,win_xright_high)

                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                
                # If you found > minpix pixels, recenter next window  
                if(len(good_left_inds) > minpix):
                    leftx_current = int(np.mean(nonzerox[good_left_inds]))
                if(len(good_right_inds) > minpix):
                    rightx_current = int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices (previously was a list of lists of pixels)
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

        else:
            left_lane_inds = ((nonzerox > (self.left_lane_line.solve_average(nonzeroy)-margin)) & (nonzerox < (self.left_lane_line.solve_average(nonzeroy)+margin)))
            right_lane_inds = ((nonzerox > (self.right_lane_line.solve_average(nonzeroy)-margin)) & (nonzerox < (self.right_lane_line.solve_average(nonzeroy)+margin)))
            
        # Extract left and right line pixel positions
        self.left_lane_line.allx = nonzerox[left_lane_inds]
        self.left_lane_line.ally = nonzeroy[left_lane_inds] 
        self.right_lane_line.allx = nonzerox[right_lane_inds]
        self.right_lane_line.ally = nonzeroy[right_lane_inds]

        return

    def get_good_inds(self,nonzeroy,nonzerox,win_y_low,win_y_high,win_x_low,win_x_high):
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        return good_inds

    def fit_polynomial(self,binary_warped):
        # Find our lane pixels first
        self.find_lane_pixels(binary_warped)
        self.ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

        # Fit a second order polynomial
        self.left_lane_line.fit()
        self.right_lane_line.fit()

        # For all y values in ploty solve and get the x values
        left_fitx = self.left_lane_line.solve(self.ploty)
        right_fitx = self.right_lane_line.solve(self.ploty)

        # store the line in the line class
        self.left_lane_line.append_line(left_fitx,self.ploty)
        self.right_lane_line.append_line(right_fitx,self.ploty)

        self.lost = False

        return

    def process_image(self,img):

        # undistort image based on the loaded camera calibration data
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

        # color transforms, gradients or other methods to create a thresholded binary image
        hls_binary,hls_color = hls_gradient_select(undist)
        #cv2.imwrite( "output_images/test4_gradient.jpg", hls_binary.astype('uint8') * 255)
        #cv2.imwrite( "output_images/test4_gradient_colortest.jpg", hls_color)
        
        # Do a perspective transform
        binary_warped = cv2.warpPerspective(hls_binary,self.M,(img.shape[0],img.shape[1]))
        
        # find and fit lane line
        self.fit_polynomial(binary_warped)

        ## Visualization ##
        vis_img = np.zeros([binary_warped.shape[0],binary_warped.shape[1],3],dtype=np.uint8)

        # Colors in the left and right lane regions
        left_line = np.array([self.left_lane_line.bestx, self.ploty], dtype=np.int32).T
        right_line = np.array([self.right_lane_line.bestx, self.ploty], dtype=np.int32).T
        both_lines = np.concatenate((left_line, np.flipud(right_line)), axis=0)
        cv2.fillPoly(vis_img, [both_lines.astype(np.int32)], (255, 255, 0))
        cv2.polylines(vis_img, [right_line.astype(np.int32)], False, (255, 0, 0),thickness=5 )
        cv2.polylines(vis_img, [left_line.astype(np.int32)],False,  (0, 255, 0), thickness=5)

        # transform the lane lines back into real space, before adding text
        out_img = cv2.warpPerspective(vis_img, self.M, img.shape[::-1][1:3], flags=cv2.WARP_INVERSE_MAP)

        # Add Curvature of lane and the distance from the center of the road to the center of the car
        average_curvature = np.mean([self.left_lane_line.radius_of_curvature,self.right_lane_line.radius_of_curvature])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out_img, "Road curvature: {:6.2f}m".format(average_curvature), (700, 50), font, fontScale=1, thickness=2, color=(255, 255, 255))
        meters_from_center = measure_distance_to_center(self.left_lane_line.solve_average(binary_warped.shape[0]),self.right_lane_line.solve_average(binary_warped.shape[0]),binary_warped.shape[1])*self.left_lane_line.xm_per_pix
        cv2.putText(out_img, "Road center distance: {:6.4f}m".format(meters_from_center), (700, 100), font, fontScale=1, thickness=2, color=(255, 255, 255))
        
        # add visualization to the real image
        out_img = cv2.addWeighted(img, 1, out_img, 0.5, 0)

        # Show the warped lane line on top of the gradient image
        if self.debug:
            binary_warped_color = cv2.warpPerspective(hls_color,self.M,(img.shape[0],img.shape[1]))
            binary_warped_color = cv2.addWeighted(binary_warped_color, 1, vis_img, 0.3, 0)
            binary_warped_color = cv2.resize(binary_warped_color,img.shape[::-1][1:3])
            out_img = np.concatenate((out_img, binary_warped_color), axis=0)


        return out_img

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
    parser = argparse.ArgumentParser(description='Maps out the car lane in a video stream')
    parser.add_argument('--i', dest='input',
                        help='Video file to add lane line to',
                        default='project_video.mp4', type=str)
    parser.add_argument('--o', dest='savePath',
                        help='path to save the video',
                        default="output_images/project_video.mp4", type=str)
    parser.add_argument('--c', dest='calibrationFile',
                        help='Camera calibration file, can be created with calibrate_camera.py',
                        default='camera_calibration.p', type=str)
    parser.add_argument('--d', dest='debug',
                        help='if used add debug video output',
                        default=False, type=bool)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    print ('Called with args:')
    print (args)

    fll = FindLaneLines(args.calibrationFile,args.debug)
    

    clip1 = VideoFileClip(args.input)
    white_clip = clip1.fl_image(fll.process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(args.savePath, audio=False)

    # read in the camera calibration
    #images_dir = 'test_images/test*.jpg'
    #images_dir = 'test_images/straight_lines*.jpg'
    #images = glob.glob(images_dir)

    #for fname in images:
        #img = cv2.imread('test_images/test4.jpg')
        #out_img = fll.process_image(img)
        #cv2.imwrite( "output_images/test4_final.jpg", out_img)
        #plt.imshow(out_img)
        #plt.show(block=False)
        #input('press <ENTER> to continue')
