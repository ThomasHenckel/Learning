#Advanced Lane Finding Project

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Distorted"
[image2]: ./output_images/calibration1_undistorted.jpg "Undistorted"
[image3]: ./output_images/test4_undistorted.jpg "Undistorted_road"
[image4a]: ./output_images/test4_gradient.jpg "Binary Example"
[image4b]: ./output_images/test4_gradient_color.jpg "Binary Example color"
[image5]: ./output_images/straight_lines1_warplines.jpg "Unwarp Example"
[image6]: ./output_images/straight_lines1_warped.jpg "Warp Example"
[image7]: ./examples/color_fit_lines.jpg "Fit Visual"
[image8]: ./output_images/test6_final.jpg "Output"
[image9]: ./output_images/debug_output.jpg "Output"
[image10]:./test_images/test4.jpg "Original"
[image11]: ./examples/curve.jpg "Fit Visual"
[video1]: ./output_images/project_video.mp4 "Video"

### Summary
To try out the lane finding there are two steps, camera calibration using calibrate_camera.py, it takes checboard calibration images, and creates a camera calibration file needed for next step
Run: __python calibrate_camera.py --i camera_cal/calibration*.jpg --o camera_calibration.p --nx 9 --ny 6__
When camera calibration is done find_laneline.py can be used to detect lane line in a video stream, the output is the original video with the detected lane, the script also calculates the curvature and the distance between center of the car to center of the road.
run: __python find_laneline.py --i project_video.mp4 --o output_images/project_video.mp4 --c camera_calibration.p --d False__
[This will produce an output like this](./project_video.mp4)
The --d will addthe input to the lanefinder to the video input, this is usefull to understand what information the system has to work with after the image manipulation

![alt text][image9]

Below is a in depth description of the process including the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Camera Calibration

The code for this step is contained in [calibrate_camera.py](calibrate_camera.py), the code takes a folder with checkboard images, and do the camera calibrations and saves the camera matrix, distortion coefficients to a file.
run: __python calibrate_camera.py --i camera_cal/calibration*.jpg --o camera_calibration.p --nx 9 --ny 6__

Code start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. The object points is the location of the chesboard corners in a flat plain, as the checkboard corners are evenly spaces, and on a flat plane, the object points is an array [0., 0., 0.] to [8., 5., 0.]. the object points does not need to scale to the original image size and can therefor just be the relative location of the corners starting from topleft counting from left to right and top to bottom.

For each image the corners of the image is found using `cv2.findChessboardCorners()` if all nx and ny corners are detected the (x,y) cordinates are added to the `imgpoints` array
When corners are found for all the images the camera calibration is done with `cv2.calibrateCamera()` that gives us the camera matrix, distortion coefficients we need to undistort images using `cv2.undistort()`

Original Image             |  Undistorted image
:-------------------------:|:-------------------------:
![alt text][image1]  |  ![alt text][image2]

### Pipeline
This section describes the pipeline from raw image input, to the final resoult that is the detected lane placed as an overlay to the original image. the code for this is found in [find_lanelines.py](find_lanelines.py)

#### 1. Distortion-corrected.

The input image is corrected for distortion by taking the camera matrix, distortion coefficients found in the camera calibration, and applying cv2.undistort() to the image.  
The difference is minor looking at it, but it important when we later warp the image.


Original Image             |  Undistorted image
:-------------------------:|:-------------------------:
![alt text][image10]  |  ![alt text][image3]

In code see first line of `process_image()`

#### 2. Color transforms and gradients

The goal here is to get a binary image of the road, where the lane lines stand out as much as possible and is robust enough to deal with lane lines of different colors while also being robust to various lighting conditions.
The idear and code for this transformation is from udacity self driving car course, from the lesson "Gradient and color spaces"
instead of using the grayscale, or a RGB mask this aproach uses the HLS color representation of the image. it turns out that the S representing the light intensity and the S representing the saturation are really good when destinguising lanelines from the asfalt. we will not be using the H parameter that stands for Hue or color, and it should therefor not matter if a laneline is white or yellow.
This approach is combined with Sobel edge detection with the kernel set to look for horizontal lines

To get the binary image a thresholds was applied to the output output of the S and H channel.
On the images below it is clear that the H (green) channel is unable to detect the line on the light gray asfalt, but in combination with the S (red) it is possible to create a binary image with the laneline clearly visible.

H and S channel seperatly             |  Binary image
:-------------------------:|:-------------------------:
![alt text][image4b]  |  ![alt text][image4a]

In code see first `hls_gradient_select()`

#### 3. Perspective transform

To get a top view of the lane a perspective transform is done. the source points for this transform is found by looking at [straight_lines1.jpg](test_images/straight_lines1.jpg).
This image has straight lane lines, and the source points are found by simply mapping out the section of the road that we would like to use in our lane finding later in the pipeline.
When finding the source ponts we made sure to have the top and the buttom cordinates at the same hight, later the cordinates in the right side were corrected to have the same distance to the edge of the picture as the left.
The last correction is as it is stated that the camera is placed in the center of the car, and we later want to use the offset from the center to each lane line to calculate the cars offset from the center of the lane.
The source point are a 1280x720 image, where we place the lane with a 150px padding. If there were no padding the top of the lanes would disapear in bends, or if the care is a bit f from the center of the road. but the transformed image also works as a mask hising and keeping the padding low hides unwanted objects from the field of view. 

The transformation vector is created in `get_perspective_transform_vector()` and applied to the images in the pipeline in the 3. line of `process_image()`

```python
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
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 588, 455     | 150, 0        | 
| 257, 682      | 150, 1280      |
| 1023,682    | 570, 1280      |
| 692,455      | 570, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. 

Unwarped image            |  Warped image
:-------------------------:|:-------------------------:
![alt text][image5]  |  ![alt text][image6]

Code see here [generate_writeup_images.py](generate_writeup_images.py).

#### 4. Find and fit lane lines
1. Finding lane line pixes has one process for cold start, and one when line was allready detected in the previous frame. Cold start process is also used if the lane line is lost, and have to be found again.
__Cold start__
Finding the lane line starts by taking the lower half of the image and in the horizontal direction summig up all activadet(white) pixels. this gives us a histogram where the peaks should be the lanelines.
Taking the peak on the right and left half of the image gives us a starting point for our lane detector.
From the starting point the line is detected using a sliding window, with a predefined hight and with. for each time the window is slided up, we move the center of the window to the mean of the x position of all the active pixels in the window. 
The sliding window is only moved if we are confident that we have found a line, that is if the number of active pixels is over some threashold. Thil last rule makes sure that our window does not run of in some random direction when detecting dashed lines. but it also may make it impossible to start the line detector on a curved dashed line.
__Find line from previous frame__
When the lane line is allready found, we can look for it based on the previous fitted line, this is a simple process of taking all active pixels with some defined margien of the last line. to make this aproach a bit more robust the search arear is not actually the last prediction, but instead the average of the last 10 predictions, smoothening out lines that might be a bit off.
lane finding is performed in `find_lane_pixels()`
2. When a set of activated pixels are found, the X and Y pixel locations are fitted using `np.polyfit()` this fit has x and y reversed as the lane line might have the same y value for several x values.
When a line is fitted, the x values for each y value in the image is calculated, and this is the line used to display the found lane line.
To compensate for the jitter each fitted line is added to a sliding average, and the calculated x values is as well added to a sliding average, to smothen the display of the lane line.
The code for fitting and averaging is in the `Line()` class

![alt text][image7]

#### 5. calculating radius of curvature and vehicle offset with respect to center.

calculating the radius on a given point of a polynomial can be expressed by the formal seen below. so to find the radius of curvature for our lane line we just need to take solve the line polynomial for some y.

![alt text][image11]
formel thanks to https://www.math24.net/curvature-radius/

however solving the equation would give us the radious in warped image space in pixels, to get the distance in meters we first need to know the conversion between warped images space and real world space. The conversion is derived from two assumptions:
1. The lane width is 3.7 meter
2. The length of a dashed lane line is 3 meter

By taking the width between the lane lines and the length of a lane line in a warped image we get the following convertion
```python
ym_per_pix = 3/290 
xm_per_pix = 3.7/500 
```
The conversion could properly be used on the pixel space radious, given that we know its orientation in respect to x and y, but another way that is used in this code is to just convert each x and y to real space and fit and then use this new polinomical in real space to find the distance.
The code for calculating the distance is in `measure_curvature()` and `calculate_curvature()` the calculation is performed in the Visualization step and takes `bestfitx` (average x over 10 lines) using the average of left and right. 


#### 6. Lane plot on original image

Finally the identified lane lines and the area between the lines are plotted onto the warped image, this is then warped back and combined with the original image. text are added to the final unwarped image.

![alt text][image8]

code is is found in `process_image()`

if debug is selected, the laneline is added to the warped image with seperate H and S channels.

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
