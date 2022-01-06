# **Finding Lane Lines on the Road** 
[//]: # (Image References)

[image1]: ./images/grayscale.jpg "Grayscale"
[image2]: ./images/blur.jpg "Blur"
[image3]: ./images/edges.jpg "Edges"
[image4]: ./images/hough_line.jpg "Houge line"
[image5]: ./images/lane_lines.jpg "Lane Line"


## Reflection
As part of this first mini project we use simple image processing steps like edge detection and Hough Line, to find lane lines on the road. The algorithm relies on some domain knowledge about where to expect the right and left lane line to be, and uses a region mask to ignore lines found in other parts of the image.
Final processing step is to combine the found lines to make one solid line in each side, This line should as close as possible correspond with the real lane lines in the image.
This algorithm only tries to fit a straight line, and can works on straight roads, but will have challenges in curves.

## 1. Processing Pipeline

### 1. Grayscale
We can grayscale the image, as the light intensity between the pavement and both white and yellow lane lines is significant enough to detect them

![alt text][image1]

### 2. Blur
Blurring the image removes noise, and makes it easier to detect the edges from the lane lines

![alt text][image2]

### 3. Canny edge detection
Some tuning is needed here to find the lane lines, while ignoring edges in the image that is not part of the lane line

![alt text][image3]

### 4. Hough line
from all the edges consistent lines are found, here some parameter tuning is also needed, to find consistent lines that correspond to lane lines, to improve the result this step uses a region of interest mask, to only find lines in the region of the image where we expect the lane line to be.
This step is done once for the right and once for the left lane line

![alt text][image4]

### 5. Fit the lines to one line
for both the right and left lane line several line segments are found by the Hough line. both the solid lane line in one side of the road and the dashed in the other side, is at this stage represented by several lines.
To get the solid line each line segments start- and end coordinates are represented by a point in the image, and a linear fit is used to find the line that best represents these points.

![alt text][image5]

### 2. Identify potential shortcomings with your current pipeline

1. pipeline step 1-4 relies on some trail an error parameter tuning, and it is unknown how well this approach will work in different light settings

2. providing a region of interest, is problematic for lane changes, or if the road is turning so much that the lane line is outside or interest area

3. Fitting the starting and ending point of all lines found is problematic in several ways
- Long lines and short lines have the same importance, so just a couple of short "outlier" lines can have a big impact of the fitted line
- lines that is not oriented in the direction we expect the lane line to be, is also included when we do the line fit. in "challenge.mp4" the lower part of the image shows the hood that provides a lot of edges that influences the lane line detection. 


### 3. Suggest possible improvements to your pipeline

The first thing i would try out if I had to improve the pipeline would be to sort out lines from the fit function that does not have the general direction we expect the lane line to have.

Then I would try to make the weight of the line be relative to its length, either with a weight parameter of by simply adding points along the line

Further improvements could be to use results and image data from prior image frames, this could make the lane finding less likely to jump totally of for one or two frames. it could also help stich together the dashed lane lines, and remove false edge detections, as we know in which direction an edge ought to move from frame to frame.
