# **Behavioral Cloning** 

### Summary

This project shows how to train a convolutional network using images of the road and the steering angle. in the project we use a [simulator](https://github.com/udacity/self-driving-car-sim) both to generate training data, and later to test the generated model.

Generate training images:
Start up the simulator in "Training mode" press 'r' whenever you want to record, and then drive the car around the track keeping it fairly centered on the road. this will produce a folder with images.

Training the network:
run: python model.py -i path/to/images
This will train for 5 epochs, to continue training add -m model.h5

Test out the model on the simulator:
Run: python drive.py model.h5
and then start the simulator in "Autonomous mode"

[This is](./output/run1.mp4) a video of the resulting driving by the model trained with 2 laps of training data.


[//]: # (Image References)

[image1]: ./images/center_2018_12_01_13_49_58_534.jpg "Center Image"
[image2]: ./images/flip_center_2018_12_01_13_49_58_534.jpg "Flip Center Image"
[image3]: ./images/cropped_center_2018_12_01_13_49_58_534.jpg "Cropped Image"
[image4]: ./images/left_2018_12_01_13_49_58_534.jpg "Left Image"
[image5]: ./images/right_2018_12_01_13_49_58_534.jpg "Right Image"
[image6]: ./images/cnn-architecture-624x890.png "CNN Architecture"
[image7]: ./images/loss_plot.png "Loss Plot"

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to take small steps, first using a simple network with only a flatten layer, and then test out the whole training validation pipeline, that consists of collection some driving data from the simulator, loading the images and do a train/test split, training a model, and finally driving the car in autonomous mode using the simulator.

When I had this loop working the following steps, mentioned in [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)
- Normalize and crop input
- Flipping images and steering angels
- Use side mounted cameras for training
- Implement the CNN mentioned in the Nvidia article
- Generate more training data

The code for the model training is found in [model.py](./model.py), the training uses a 80/20 train/test split, and callback function is used to ensure that only models with decreased validation loss is saved.

##### Normalize and crop input
The most common normalization on images is to divide by 255 to get the input in a range from 0-1, this I have until now done as a preprocessing step, but Keras also have possibility to add this into the model as shown in the first layer of the model in the function train_model_dataframe() `Lambda(lambda x: x / 255.0)`
Another step that can also be performed in the model is cropping the image, in our case the bottom part of the image only display the hood of the car, and the top part only objects surrounding the road, by cropping these distractions away we expect to get better accuracy.

Original Image             |  cropped image
:-------------------------:|:-------------------------:
![alt text][image1] |  ![alt text][image3]

##### Flipping images and steering angels
The test track is a circular, and the car is driving counter clockwise on the track, because of this the car would steer more to the left than to the right, and if we don’t compensate for this our model would learn this behavior. a good way of compensating for this is flipping all images on the horizontal axis, and then also reversing the steering angle.

Original Image             |  Flipped image
:-------------------------:|:-------------------------:
![alt text][image1] <br> steering angel = -0.066 |  ![alt text][image2] <br> steering angel = 0.066

Flipping of images and adding the flipped images to the training data is done in flip_images() and use_flipped_images()

##### Use side mounted cameras for training
Producing training data that lets the model learn to take the model to the center of the road can be hard to produce, if driving perfectly centered on the road no data is there to show what action to take when the car is getting of the center. One way to get this data is to record data starting the car off the center and driving it back to the center of the road. another approach is to use the side mounted cameras, and then adding a bit to the steering angle on the left image and subtracting a bit from the right.

The steering angle compensation to use in the final model was 0.15, initially it was set to 0.2, but this made the car oscillate on some of the straight parts. lowering it to 0.1 made this a bit better, but resulted in the car missing a turn and driving into the water. so in the end 0.15 was chosen.

left Image             |  Center image|  Right image
:-------------------------:|:-------------------------:|:-------------------------:
![alt text][image4] <br> steering angel = 0.084 |  ![alt text][image1] <br> steering angel = -0,066|  ![alt text][image5] <br> steering angel = -0,216

Adding side camera images to the training data is done in use_side_camera() note that this is done after flipping, so we also get flipped images of the side cameras.

##### Implement the CNN mentioned in the Nvidia article
In [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) Nvidia shows the network architecture that they have successfully used to train a self-driving car, it consists of a normalization layer followed by 5 convolutional layers, and then 4 fully connected layers.

![alt text][image6]

The first 3 convolution layers has a 5x5 kernel and uses a stride of 2x2 and the last 2 use a kernel of 3x3 and a 1x1 stride.

This network architecture was actually also the final architecture, as the next step of generating 2 full laps of training data, was able to bring the care safely through the training track.

The model architecture and the training is found in train_model_dataframe()

##### Generate more training data
For each of these steps until this point the training data was only around 100 meters of the test track but was enough to show that each step was improving the model.
generating more training data was done by taking the car around the track 2 full rounds, keeping the care as good as possible in the center. this was done using the mouse and was not all that simple, i have generated a small vide of my [driving skills.](./output/run0.mp4)
I tend to hug the inside of the curves, but my theory is that it is not all that bad, as the car was always more likely to drive on in a curve than turning to much.

##### Train the model, and testing it out on the track
The final model were trained on 22213 training images, taken from 2 laps of driving around the track.
while training the validation loss was observed, and the model was saved after each epoch, if the validation loss decreased from the previous epoch.
The important thing to watch for is the difference in validation and training loss, if the training loss starts to decrease, while the validation loss stops decreasing or increases, that is a sign of overfitting, and the training can stop.

The figure below shows training for 20 epochs

![alt text][image7]

After 15 epochs the validation loss starts to flatten out. And the best model is saved at Epoch 20.
A test of this model shows that it is taking the care successfully around the track, even when setting the speed to 30mph. [See the video here](./output/run1.mp4)
We could properly decrease the loss a bit more by continue training, as the validation loss are slightly decreasing, but as the car drives fine we stop training here.

If the accuracy at the point where the validation loss stops decreasing is not good enough, there are several things that can be tried out like generating more data and try out dropout layers in the model architecture.

#### 2. Future work
In the simulator there are a 2. track, where the lane lines are different, and where the lane is intended for traffic in both directions.

The model fails completely driving the car on this track, and it looks like the model trained for 20 epochs was worse than the same model trained on 5 epochs.

Here I think we see a overfitting problem, although the model is not overfitting that much when looking at the training and testing loss, it is overfitting to the track, as all the images are from the same track, and those 2 laps on the track.

So it will be fun to see how well the care does then having training data from the 2 tracks, or if it is actually possible to generalize the model with data from track 1 so much that it can run on track 2.
