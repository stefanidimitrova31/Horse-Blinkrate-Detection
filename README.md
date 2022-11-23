# Emily-Horse-Blink

This repository hosts all code necessary for the Horse Blink Project.

## Setting up:

You need to create a folder on your Desktop called Horseblink. Inside, create three folders: code, videos, data. You want to put the file called Emily_completed.py file in the folder called 'code'. In the videos folder make sure to put all of your Horse videos. The data folder is where the CSV files will be saved once you run the code for a video.

## System requirements:

In order to run the code you need, a distribution of the scientific environment for Python - Spyder (distributed by Anaconda) and the following libraries/modules. Some of them will already come with your distribution of Spyder so it is a matter of just importing them:

* opencv-contrib-python
* tkinter
* numpy
* matplotlib
* csv
* pandas
* datetime (timedelta)

## Running the algorithm:

* User unput:

Before you run the algorithm, there are a few parameters that rely on user input. If you don't change these parameters, you will be prompted to select a video and the program will run for the entirety of its duration. The parameters taking user input are:

```
startframe
```
In case you wanted to run a specific section of the video you can change this variable to the frame number you would like to start from. The default value of this is 1 (first frame).

```
stopframe
```
In case you wanted to run a specific section of the video you can change this variable to the frame number you would like to stop at. The default value of this is last frame - 10 frames (the algorithm often buffers on the very last frame hence the measure)

```
desiredvideo
```
This variable takes input of either 0 or 1, respectively 0 meaning that you would like to run a video anew, 1 meaning that you wish to plot the signal of a video that you have ran before. Upon selecting 0 a prompt window will open where you will chose your video you would like to run. Upon selecting 1, a prompt window will open where you should select the *file_name* Signal.csv of the video you would like to run. The default value of this is 0.

The algorithm won't limit you to 60 frames, but it is advised that you run at least 60 frames of a video in order for the background substraction algorithm to be accurate. This is a warning that will show up in the console if you chose to run a video for less than 60 seconds.

```
The graph
```
Once the video is ran, a graph with the signal will be plotted. On this graph you are to chose the threshold. Upon clicking on the graph a red line will appear to illustrate where you have placed the threshold (you should attempt to click at your desired amplitude).

* Outputs:

Once you have selected your threshold, the rest of the program will run within seconds and your outputs will be ready.

* **file_name** + Variables.csv - This file contains the variables you have used for the video: startframe, stopframe, amp_threshold, frame_threshold
* **file_name** + Signal.csv - This file contains the Raw, filtered and background signal for each frame of a video that you have ran, plus the coordinates of the selected ROI (region of interest)
* **file_name** + Evaluation.csv - This is the file with the highest importance to you. It contains the frames of the blinks (start and stop frame), the duration in microseconds and the average interval between blinks throughout the video.



