# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

[//]: # (Image References)
[image1]: ./images/path_planning.gif "run1"

The goal of this project is to steer a care, given its offset from the center of the road CrossTrackError(cte). The project uses a [simulator](https://github.com/udacity/self-driving-car-sim/releases/tag/v1.45) that provides the cars cte, and this program then calculates the appropriate steering angle.

To produce a smooth trajectory for the car a PID controller is used to calculate the steering angle from the cte.
A PID controller consists of 3 parts:
- P: makes sure to steer the car towards the center of the road, proportionally to how far it is from the center.
If the car would only steer in proportion to its offset to the center, the car would start to oscillate, as it would steer its wheels towards the center, until the time it reaches the center, at this time the car would not be parallel to the center line of the road, and would therefore continue, overshooting, and starting to steer the car back to and over the center again.
- D: The Delta part of the controller counter steers, and as the P weakens as the car approaches the center of the road, the D controller makes sure that the speed towards the center is dampened. the delta is the current cte minus previous cte
- I: The I part of the controller centers the car when driving around curves. with only the P and D controller the car would always drive off the center in curves. the I is the sum of all the CTE, and pushes the car towards the center of the road in curves.

**Tuning the parameters of the PID Controller**
Each part of the PID controller is multiplied with a constant factor. Finding these 3 constants is most times a bit of a trail and error process. Helping with this trial and error the Twiddle algorithm is proposed in one of the lessons.
Using the Twiddle algorithm on some random initial values would take quite some time, so to start with some initial values that could drive the car to the first bend was selected with manual trial and error, starting by setting only the P and D parameters.
After the Twiddle algorithm was set to work.

## Running the Code.

### Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`.

Start up the simulator

### Simulator.
You can download the Term2 Simulator which contains the Path Planning Project from the [releases tab](https://github.com/udacity/self-driving-car-sim/releases/tag/v1.45).  

To run the simulator on Mac/Linux, first make the binary file executable with the following command:
```shell
sudo chmod u+x {simulator_file_name}
```

---

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1(mac, linux), 3.81(Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `./install-mac.sh` or `./install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Simulator. You can download these from the [project intro page](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.

Fellow students have put together a guide to Windows set-up for the project [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/files/Kidnapped_Vehicle_Windows_Setup.pdf) if the environment you have set up for the Sensor Fusion projects does not work for this project. There's also an experimental patch for windows in this [PR](https://github.com/udacity/CarND-PID-Control-Project/pull/3).


