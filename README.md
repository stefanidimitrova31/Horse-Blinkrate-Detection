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
In case you wanted to run a specific section of the video you can change this variable to the frame number you would like to start from.

```
stopframe
```
In case you wanted to run a specific section of the video you can change this variable to the frame number you would like to stop at.

```
desiredvideo
```
This variable takes input of either 0 or 1, respectively 0 meaning that you would like to run a video anew, 1 meaning that you wish to plot the signal of a video that you have ran before. Upon selecting 0 a prompt window will open where you will chose your video you would like to run. Upon selecting 1, a prompt window will open where you should select the *file_name* Signal.csv of the video you would like to run.

The algorithm won't limit you to 60 frames, but it is advised that you run at least 60 frames of a video in order for the background substraction algorithm to be accurate. This is a warning that will show up in the console if you chose to run a video for less than 60 seconds.



