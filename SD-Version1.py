# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 12:03:20 2022

@author: 44736
"""

#import libraries

import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from statistics import mean
from scipy import signal
import matplotlib.lines as lines
import random
from datetime import timedelta


#global variables

ROIsize = 100 # area of interest size, so the window doesn't shrink below it
BGRwinsize = 60 # subtraction window size
BGRwinoffset = int(BGRwinsize/2)-1 # 29

# USER INPUT REQUIRED

# For desired video - chose 0 if you want to run a new video, 1 if you would like to repeat a video with saved signal.
# If you chose one the program will prompt you to chose a Signal file and plot it so you can specify your threshold

desiredvideo = 0

# Input your desired start frame
startframe = 1

# If you wish to change the stop frame (otherwise video will roll until the end), go to line #and specify it there
stopframe = 0


# threshold for minimal amount of frames between two blinks, used for merging
frame_threshold = 20
global totalframenumber
global frame_inner
fps = 60


# The script requires at least 30 frames to perform good background subtraction, this is a warning that will come up.
if stopframe < BGRwinsize:
    print('It is advised to run the video for at least 60 frames, otherwise algorithm will not perform good background substraction')

#font specification
font = cv2.FONT_HERSHEY_SIMPLEX;
org = (50, 50);
fontScale = 0.5;
font_color = (255, 255, 0);
thickness = 2;

#initializing tracker
tracker = cv2.TrackerCSRT_create();
backup = cv2.TrackerCSRT_create();

#function that allows us to select file
def openvideofile():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()
    return file_path


#run video frame by frame
def framebyframe(file_path):
   
    #A = Average of the BGR channel for the selected area of interest
    
    global fps
    framecount = 0
    framenumber = []
    B = []
    G = []
    R = []
    A = []
    
    if desiredvideo==1:
        root = tk.Tk()
        csv_file_path = filedialog.askopenfilename()
        root.withdraw()
        temp = pd.read_csv(csv_file_path, delimiter =',')                    
        Avfiltered = temp['Filtered signal'].tolist()
        roi = temp['ROI'].tolist()
        print(type(Avfiltered))
    else:
        cap = cv2.VideoCapture(file_path)
        if (cap.isOpened()== False):
            print("Error")
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, startframe)
        totalframenumber = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
        
        # runs the video until the last frame (totalframenumber-10) unless otherwise specified
        stopframe = 500
        print("Total frame number", totalframenumber)

        if (cap.isOpened() == True):
            ret, frame = cap.read()
            if ret == True:
                #increase framecount with 1 after each frame
                framecount = framecount+1
                framenumber.append(framecount)
               
                if desiredvideo==0:
                    roi = cv2.selectROI(frame);
                else:
                    root = tk.Tk()
                    csv_file_path = filedialog.askopenfilename()
                    root.withdraw()
                    temp = pd.read_csv(csv_file_path, delimiter =',')                    
                    roi = temp['ROI'].tolist()
                    
                frame_cropped = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0] + roi[2]), :]
               
               
                tracker.init(frame, tuple(roi));
                backup.init(frame, tuple(roi));
               
                channelB, channelG,channelR = cv2.split(frame_cropped);
                B.append(np.mean(channelB))
                G.append(np.mean(channelG))
                R.append(np.mean(channelR))
                A.append((B[-1] + G[-1] + R[-1])/3)
       
        #while video is running for each frame update the tracker based on the area of interest
        while (cap.isOpened()):
           
            ret, frame = cap.read()
            if ret == True:
                ret, roi = tracker.update(frame);
            else:
                print("Buffering on frame:", framenumber[-1])
           
            if ret == True:
                framecount = framecount+1
                framenumber.append(framecount)
    
    
                #get center of roi and use it create an inner frame with size 50x50
                roi = list(roi)
                if roi[1]<0:
                   roi[1] = 1
                frame_cropped = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0] + roi[2]), :]
                c1 = int(roi[3]/2)
                c2 = int(roi[2]/2)
               
                # adjust inner frame size
                if c1 < ROIsize:
                    c1 = ROIsize
                   
                if c2 < ROIsize:
                    c2 = ROIsize
                   
                frame_inner = frame_cropped[c1 - ROIsize:c1+ROIsize, c2 - ROIsize:c2+ROIsize, :]
               
                # take channel averages for the inner frame
             
               
                channelB,channelG,channelR = cv2.split(frame_inner)
    
                B.append(np.mean(channelB))
                G.append(np.mean(channelG))
                R.append(np.mean(channelR))
               
               
                A.append((B[-1] + G[-1] + R[-1])/3)
               
                frame_cropped = cv2.circle(frame_cropped,(c2,c1), 10, (0,0,255), -1)
    
               
                frame_cropped = cv2.putText(frame_cropped, "Frame number: " + str(framecount), org, font, fontScale,
                        font_color, thickness, cv2.LINE_AA);
    
                cv2.imshow('Frame', frame_cropped)
                cv2.imshow('Inner frame', frame_inner)
    
               
                if cv2.waitKey(1) & 0xFF == ord('Q'):
                    break
               
                if stopframe > 0:
                    if framenumber[-1] >= stopframe:
                        break

        cap.release()
        cv2.destroyAllWindows()
    return A, framenumber, roi, stopframe
              
# once the video is ran and the raw signal is obtained this function takes the signal and filters it into background subtraction
# and filtered signal

def substractBackground(A):
    Abg = np.convolve(A, np.ones(BGRwinsize), 'valid') / BGRwinsize
    Abg = np.pad(Abg, (BGRwinoffset, BGRwinoffset+1), 'constant', constant_values = (0,0))
    Af = A - Abg # originally 29; -30
    return Af, Abg

# This function converts the filtered signal to binary (potential blinks) based on a threshold specified by the used on the graph

def convertBinary(Af, amp_threshold):
    arraysbinary = []
    for index, value in enumerate(Af):
        if value < amp_threshold:
            arraysbinary.append(1)
        else:
            arraysbinary.append(0)
    return arraysbinary

# This function detects blinks by their Start and End frames
def detectBlinks(framenumber, AfBin):
        # blinkCounts = 0
        blinkStartFrame = []
        blinkStopFrame = []
        for n in range (0, len(AfBin)-1):
            if AfBin[n] == 0:
                if AfBin[n+1] ==1:
                    # blinkCounts = blinkCounts + 1
                    blinkStartFrame.append(framenumber[n+1])
            if AfBin[n] ==1:
                if AfBin[n+1] == 0:
                    blinkStopFrame.append(framenumber[n])
               
        return blinkStartFrame, blinkStopFrame
   
# This function merges blinks if they are too close to each other (frame_threshold)
def blinkMerge(blinkStartFrame, blinkStopFrame):
    i = 0
    while i <len(blinkStartFrame)-1:
    # for i in range(0, len(blinkStartFrame)-1):
        if blinkStartFrame[i+1] - blinkStopFrame[i] < frame_threshold:
            print(blinkStartFrame[i+1])
            print(blinkStopFrame[i])
            blinkStopFrame[i] = blinkStopFrame[i+1]
            del blinkStartFrame[i+1]
            del blinkStopFrame[i+1]
        i = i+1
    return blinkStartFrame, blinkStopFrame

# This function refines the binary signal according to the new, merged start and stop frames
def createBinary(framenumber, blinkStartFrameMerged, blinkEndFrameMerged):
    binarySignal = []
    print(len(framenumber))

    for i in range(0, len(framenumber)):
        flag = 0
        for g in range(0, len(blinkStartFrameMerged)):
            if i >= blinkStartFrameMerged[g] and i<= blinkEndFrameMerged[g]:
                binarySignal.append(1)
                flag = 1
        if flag == 0:
            binarySignal.append(0)
       
    return binarySignal
        
# This function goes through the list of blinks and calculates in milliseconds their duration
def estimateDuration(blinkStartFrame, blinkStopFrame):
    blinkDuration = []

    for i in range(len(blinkStartFrame)):
        frameCount = blinkStopFrame[i]-blinkStartFrame[i]
        td = timedelta(seconds=(frameCount / fps))
        print(td)
        blinkDuration.append(td)

    return blinkDuration

# This function calculates the interval between two blinks (mean interval)
def estimateAverageBlinkInterval(blinkStartFrame, blinkStopFrame):
   
    estimateAverage = []
   
    for i in range(0, len(blinkStartFrame)):
        print(blinkStartFrame[i])
        print(blinkStopFrame[i-1])
        print(blinkStartFrame[i] - blinkStopFrame[i-1])
        estimateAverage.append((blinkStartFrame[i] - blinkStopFrame[i-1]))

    return mean(estimateAverage)

# This function saves the start and stop frames, the blink intervals and the durations
def saveEvaluation(bStartFrameMerged, bEndFrameMerged, blinkInterval, bDuration):
   
        fileName = filePath.split("/")[-1]
        outputName = fileName.split(".")[0] + '_Evaluation.csv'
        pathName = filePath.split("/videos")[0] + '/data/'
       
        with open(pathName + outputName, 'w', newline='') as file:
           
            writer = csv.writer(file)
           
            writer.writerow(["Blink start frame", "Blink end frame", "Est. blink interval (in frames)", 'Duration'])
            writer.writerow([bStartFrameMerged, bEndFrameMerged, blinkInterval, bDuration])

# This function saves te start, stop frame and amplitude threshold chosen by the user and the frame threshold
def saveVariables(startframe, stopframe, amp_threshold, frame_threshold):
    
    fileName = filePath.split("/")[-1]
    outputName = fileName.split(".")[0] + '_Variables.csv'
    pathName = filePath.split("/videos")[0] + '/data/'
   
    with open(pathName + outputName, 'w', newline='') as file:
           
            writer = csv.writer(file)
           
            writer.writerow(["Start Frame","Stop Frame","Amplitude Threshold", "Frame Threshold"])
            writer.writerow([startframe, stopframe, amp_threshold, frame_threshold])

# This function saves the Raw, Filtered and Average signal and the coordinates for the region of interest in case the user
# would like to repeat the experiment
def save_signal_roi(Av, Avfiltered, Abg, roi):
   
        print(filePath)
        r = zip(roi)
        fileName = filePath.split("/")[-1]
        outputName = fileName.split(".")[0] + 'Signal.csv'
        pathName = filePath.split("/videos")[0] + '/data/'
               
        with open(pathName + outputName, 'w', newline='') as file:
           
            writer = csv.writer(file)
           
            writer.writerow(["Raw signal", "Filtered signal", "Background signal"])
           
            for i in range(0, len(framenumber)):
                           
                writer.writerow([ Av[i], Avfiltered[i], Abg[i]])
                
if __name__ == "__main__":

   
    filePath = openvideofile()
    Av, framenumber, roi, stopframe = framebyframe(filePath)
    Avfiltered, Abg = substractBackground(Av)

def plotData(x, y):
   
    coords  =[]
   
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y, '-')

    # Simple mouse click function to store coordinates
    def onclick(event):
        global ix, iy
        ix, iy = event.xdata, event.ydata
   
        # assign global variable to access outside of function
        nonlocal coords
       
        coords.append((ix, iy))
   
        # Disconnect after 2 clicks
        if len(coords) == 2:
            fig.canvas.mpl_disconnect(cid)
            plt.close()
   
        return coords
   
    # Call click func
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
   
    plt.show()
   
    plt.waitforbuttonpress()
      
    return coords

e = plotData(framenumber, Avfiltered)
amp_threshold = int(e[0][1])
hor_line = plt.axhline(y=amp_threshold, color='r', linestyle='-')

# FUNCTIONS

AvfilteredBinary = convertBinary(Avfiltered, amp_threshold)
bStartFrame, bEndFrame = detectBlinks(framenumber, AvfilteredBinary)
bStartFrameMerged, bEndFrameMerged = blinkMerge(bStartFrame,bEndFrame)
print(bStartFrameMerged, bEndFrameMerged)

blinkinterv = estimateAverageBlinkInterval(bStartFrameMerged, bEndFrameMerged)
blinkDur = estimateDuration(bStartFrameMerged, bEndFrameMerged)
blinkMergeBinary = createBinary(framenumber, bStartFrameMerged, bEndFrameMerged)
variables = saveVariables(startframe, stopframe, amp_threshold, frame_threshold)
results = save_signal_roi(Av, Avfiltered, Abg, roi)
trueblinks = saveEvaluation(bStartFrameMerged, bEndFrameMerged, blinkinterv, blinkDur)
   
