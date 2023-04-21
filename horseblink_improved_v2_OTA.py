# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 14:53:59 2021

@author: Stefani Dimitrova std31
"""

#import libraries

import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import operator
import csv
import pandas as pd
from pandas import Series, DataFrame
import time
from datetime import timedelta
import statistics
#import pywt
from scipy import stats
from scipy import ndimage

# This script defines functions to perform eye-tracking analysis on video files. The main functions in this script include:

# selectHorseVideoFile(): This function opens a dialog box that allows the user to select a video file of a horse to be analyzed.

# openHorseVideoFile(videoFilePath): This function opens the selected horse video file for processing.

# extractVideoParameters(vid): This function extracts the frame rate and total number of frames in the opened video file.

# initializeEyeTracker(vid): This function initializes an eye tracker using the first frame of the video and allows the user to select the eye region to be tracked.

# selectExpertAnnotationFile(): This function opens a dialog box that allows the user to select an expert annotation file for the selected horse video file.

# openExpertAnnotationFile(expertFilePath): This function opens the selected expert annotation file.

# framebyframe(vid, eyetracker, framenumber): This function processes the selected video file frame by frame and extracts the mean and standard deviation of the eye
# region and the entire frame.

# The script also contains some additional functions for plotting and data analysis, such as plotDigitalData(), subtractBackground(), and lowPassFilter().





# -----------------------------------------------------------------------------
# Variables
# -----------------------------------------------------------------------------
vidframerate = 0
vidframenumber = 0

eyesize = 50

ampThreshold = -15

targetFrameWidth = 640
targetFrameHeight = 480
frameResize = True

def selectHorseVideoFile(): 

    print("Select a horse video file")

    root = tk.Tk()
    file_path = filedialog.askopenfilename()
    root.withdraw()
    # file_path = 'GH010031.MP4'
    return file_path

def openHorseVideoFile(videoFilePath):
    
    vid = cv2.VideoCapture(videoFilePath)
    if (vid.isOpened() == False):
        print("Error in opening video file")
    
    return vid  

def extractVideoParameters(vid):

    fr = round(vid.get(cv2.CAP_PROP_FPS))
    fn = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    return fr, fn

def initializeEyeTracker(vid):
    
    ret, frame = vid.read() 
    if ret == True:
        
        if frameResize:
            # resize the image
            frame = cv2.resize(frame, (targetFrameWidth, targetFrameHeight))
        #print(frame.shape)
        #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        #cv2.imshow(frame)
        #frame = cv2.resize(frame, (960, 720))
        cv2.namedWindow("roiframe", cv2.WINDOW_NORMAL)
        #cv2.imshow("roiframe", frame)
        #cv2.waitKey(0)
        #cv2.resizeWindow("roiframe", 960, 720)
        roi = cv2.selectROI("roiframe", frame)
        #eyetracker = cv2.TrackerCSRT_create()
        eyetracker = cv2.TrackerKCF_create()
        
        eyetracker.init(frame, roi)
        cv2.destroyWindow("roiframe")
    
    return eyetracker
    
def selectExpertAnnotationFile():    # OTAR CHANGE FUNCTION NAME
    
    print("Select the expert annotation file for the selected horse video file")
    
    root = tk.Tk()
    file_path = filedialog.askopenfilename()
    root.withdraw()
    
    return file_path

def openExpertAnnotationFile(expertFilePath):
    
    expert = pd.read_csv(expertFilePath)
    #if (expert.isOpened() == False):
    #    print("Error in opening expert annotation file")  
        
    return expert

#run video frame by frame
def framebyframe(vid, eyetracker, framenumber):
    
    
    framecounter = 0
    frameid = []
    eyemean = []
    eyesd = []
    roimean = []
    roisd = []
    #while (vid.isOpened()):
    #for k in range(0,framenumber-1,1):
    for k in range(0,1000,1):
        
        
        frameret, frame = vid.read() # read next frame
        if frameret == True:
            
            if frameResize:
                # resize the image
                frame = cv2.resize(frame, (targetFrameWidth,targetFrameHeight))
            
            trackret, roi = eyetracker.update(frame); # detect eye region
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert rgb to gray
            
            if trackret == True:
            
                cx = int( roi[1]+roi[3]/2 ) 
                cy = int( roi[0]+roi[2]/2 )
            
                roiframe = frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2], :] # crop eye region
                eyeframe = cv2.cvtColor(frame[cx-eyesize:cx+eyesize, cy-eyesize:cy+eyesize,:], cv2.COLOR_BGR2GRAY)
            
                roimean.append(np.mean(roiframe))
                roisd.append(np.std(roiframe))
            
                eyemean.append(np.mean(eyeframe))
                eyesd.append(np.std(eyeframe))
            
                framecounter = framecounter+1
                frameid.append(framecounter)
                
                print(framecounter)
            
                tempx = int(roi[2]/2)
                tempy = int(roi[3]/2)
                plotframe = cv2.circle(roiframe,(tempx,tempy), 10, (0,0,255), -1)
                plotframe = cv2.rectangle(plotframe, (tempx-eyesize,tempy-eyesize), (tempx+eyesize,tempy+eyesize), (255,0,0), 2)
            
            #frame_cropped = cv2.putText(frame_cropped, "Frame number: " + str(framecount), org, font, fontScale, 
            #        font_color, thickness, cv2.LINE_AA);

                cv2.imshow('Frame', plotframe)
                #cv2.imshow('Inner frame', eyeframe)

            
                if cv2.waitKey(1) & 0xFF == ord('Q'):
                    break
            
    vid.release()
    cv2.destroyAllWindows()
    
    return frameid, eyemean, eyesd, roimean, roisd

#NEW plot digital

def subtractBackground(A, bgrwinsize):
    bgrwinoffset = int(bgrwinsize/2)-1
    Abg = np.convolve(A, np.ones(bgrwinsize), 'valid') / bgrwinsize
    Abg = np.pad(Abg, (bgrwinoffset, bgrwinoffset+1), 'constant', constant_values = (0,0))
    Af = A - Abg   
    return Af, Abg


def convertBinary(A, th):
    Abin = []
    for index, value in enumerate(A):
        if value > th:
            Abin.append(1)
        else:
            Abin.append(0)
    return Abin

def detectBlinks(framenumber, Abin):
    blinkCount = 0
    blinkStartFrame = []
    blinkStopFrame = []
    blinkDuration = [] 
    
    flag = 0
    for n in range (0, len(Abin) - 1):
        if Abin[n] == 0:
            if Abin[n+1] == 1:
                flag = 1
                blinkCount = blinkCount + 1
                blinkStartFrame.append(framenumber[n+1])
        if Abin[n] == 1 and flag == 1 :
            if Abin[n+1] == 0:
                print(n)
                blinkStopFrame.append(framenumber[n])
                blinkDuration.append(blinkStopFrame[-1] - blinkStartFrame[-1])
                
    return blinkStartFrame, blinkStopFrame, blinkDuration, blinkCount
    
def mergeBlinks(blinkStartFrame, blinkStopFrame, blinkDuration, blinkCount, ub):
    
    blinkMergedCount = 1
    blinkMergedDuration = []
    blinkMergedStartFrame = []
    blinkMergedStopFrame = [] 
    
    blinkMergedStartFrame.append(blinkStartFrame[0]) 
    blinkMergedStopFrame.append(blinkStopFrame[0]) 
    for n in range(1, blinkCount):
        if blinkStartFrame[n] - blinkMergedStopFrame[-1] < ub:
            blinkMergedStopFrame.pop()
            blinkMergedStopFrame.append(blinkStopFrame[n])
        else:
            blinkMergedStartFrame.append(blinkStartFrame[n])
            blinkMergedStopFrame.append(blinkStopFrame[n])
            blinkMergedCount = blinkMergedCount + 1
    
    
    for n in range(0, blinkMergedCount):
        blinkMergedDuration.append(blinkMergedStopFrame[n] - blinkMergedStartFrame[n])
    
    
    return blinkMergedStartFrame, blinkMergedStopFrame, blinkMergedDuration, blinkMergedCount

def removeBlinks(blinkStartFrame, blinkStopFrame, blinkDuration, blinkCount, lb):
    
    blinkFilteredCount = 0
    blinkFilteredDuration = []
    blinkFilteredStartFrame = []
    blinkFilteredStopFrame = [] 
    
            
    for n in range(0, blinkCount):
        if blinkDuration[n] > lb:
            blinkFilteredCount = blinkFilteredCount + 1
            blinkFilteredDuration.append(blinkDuration[n])
            blinkFilteredStartFrame.append(blinkStartFrame[n])
            blinkFilteredStopFrame.append(blinkStopFrame[n])
    
    return blinkFilteredStartFrame, blinkFilteredStopFrame, blinkFilteredDuration, blinkFilteredCount



def createBinary(framenumber, blinkStartFrame, blinkStopFrame):
    binarySignal = []
    for n in range(0, len(framenumber)):
        binarySignal.append(0)
        for m in range(0, len(blinkStartFrame)):
            if n >= blinkStartFrame[m] and n <= blinkStopFrame[m]:
                binarySignal.pop()
                binarySignal.append(1)
        
    return binarySignal

def detectPeaks(framenumber, A):
    pamp = []
    pframes = []   
    for n in range(1, len(framenumber)-1):
        if A[n] < A[n-1] and A[n] < A[n+1] and A[n] < ampThreshold:
            pamp.append(A[n])
            pframes.append(framenumber[n])
    
    return pframes, pamp


def save_data(filePath):
        stefbinary = plotdigital(Af)
    
        print(filePath)
        fileName = filePath.split("/")[-1]
        outputName = fileName.split(".")[0] + '.csv'
        
        with open(outputName, 'w', newline='') as file:
            
            writer = csv.writer(file)
            
            writer.writerow(["Frame number", "Event"])
            
            for i in range(0, len(framenumber)):
                            
                writer.writerow([ framenumber[i], stefbinary[i]])
                
def differentiationAttempt(Avfiltered, interval = 1):
    diff = []
    for i in range(interval, len(Avfiltered)):
        value = Avfiltered[i] - Avfiltered[i-interval]
        print(value)
        diff.append(value)
    return diff


def compareResults(emily, blinkStartFrame):
    #import Emily's annotation and our
    flag = False
    truePositives = []
    falsePositives = []
    startblink = emily['Eye closing Frame'].tolist()
    endblink =  emily['Eye Fully Open'].tolist()
    falseNegatives = startblink.copy()

    
    # for i in range(0, len(startblink)):
    #     for i in range(0, len(endblink)):
    for index, value in enumerate(blinkStartFrame):
        flag = False
        for i in range(0, len(startblink)):
            
            if value >= startblink[i] and value <= endblink[i]:
                flag = True
                truePositives.append(value)
                falseNegatives.remove(startblink[i])
                
        if flag == False:
            falsePositives.append(value)
                    
    print("True positives: ", truePositives)
    print("False positives:", falsePositives)
    print("False negatives:", falseNegatives)
    return truePositives, falsePositives, falseNegatives, startblink, endblink

def saveEvaluation(truePositives, falsePositives, falseNegatives):
    
        fileName = filePath.split("/")[-1]
        outputName = fileName.split(".")[0] + '_Evaluation.csv'
        
        with open(outputName, 'w', newline='') as file:
            
            writer = csv.writer(file)
            
            writer.writerow(["True positives", "False positives", "False negatives"])
            maxAll = max(len(truePositives), len(falsePositives), len(falseNegatives))
            for i in range(0, maxAll):
                bList = []
                if i < len(truePositives):
                    bList.append(truePositives[i])
                else:
                    bList.append(0)
                if i< len(falsePositives):
                    bList.append(falsePositives[i])
                else: 
                    bList.append(0)
                if i< len(falseNegatives):
                    bList.append(falseNegatives[i])
                else:
                    bList.append(0)
                writer.writerow( bList)
    

if __name__ == "__main__":

    # select video file and its corresponding expert annotation file
    videoPath = selectHorseVideoFile()
    
    # open video file and its corresponding expert annotation file
    vid = openHorseVideoFile(videoPath)
    
    # extract video parameters
    vidframerate, vidframenumber = extractVideoParameters(vid)
    
    #print("frame rate = ", vidframerate, "frame number = ", vidframenumber)
    
    # initialize opencv csrt tracker
    eyetracker = initializeEyeTracker(vid)
    
    # track the eye and extract stastical features (mean and sd)
    fid, cmean, csd, rmean, rsd = framebyframe(vid, eyetracker, vidframenumber)
    
    # filter data    
    for n in range(0, len(cmean)):
        cmean[n] = cmean[n] - rmean[n]
    
    bgrwinsize = vidframerate
    cmeanfilt, cmeanbgr = subtractBackground(cmean, bgrwinsize)
    #csdfilt, csdbgr = subtractBackground(csd, bgrwinsize)
    
    # threshold data
    cmeanfiltmedian = ndimage.median(cmeanfilt[30:-29])
    cmeanfiltmad = stats.median_abs_deviation(cmeanfilt[30:-29], scale=1)
    threshold = cmeanfiltmedian + 3 * cmeanfiltmad
    cmeanfiltbin = convertBinary(cmeanfilt, threshold)
    #csdfiltbin = convertBinary(csdfilt)

    # detect peaks
    peakFrames, peakAmp = detectPeaks(fid, cmeanfilt)
    
    # wavelet
    # waveletCoeff, waveletFreq = applyWaveletTransform(fid, 1/60, cmeanfilt)
    
    # detect blinks
    bStartFrame, bStopFrame, bDuration, bCount = detectBlinks(fid, cmeanfiltbin)
    
    # merge blinks
    ub = 10
    bmStartFrame, bmStopFrame, bmDuration, bmCount = mergeBlinks(bStartFrame, bStopFrame, bDuration, bCount, ub)
    
    # remove blinks
    lb = 10
    bmrStartFrame, bmrStopFrame, bmrDuration, bmrCount = removeBlinks(bmStartFrame, bmStopFrame, bmDuration, bmCount, lb)
        
    # predicted blinks (binary signal)
    predictedBlinks = createBinary(fid, bmrStartFrame, bmrStopFrame)
    
    
    expertPath = selectExpertAnnotationFile()
    expert = openExpertAnnotationFile(expertPath)
    expertStartFrame = expert['Eye closing Frame'].tolist()
    expertStopFrame =  expert['Eye Fully Open'].tolist()
    actualBlinks = createBinary(fid, expertStartFrame, expertStopFrame)

   
        
    plt.figure()
    plt.plot(fid, cmeanfilt, 'k')
    plt.plot(peakFrames,peakAmp,'ko')
    plt.axhline(y = threshold, color = 'g')
    actualBlinks50m = [i * -50 for i in actualBlinks]
    predictedBlinks50m = [i * -50 for i in predictedBlinks]
    plt.plot(fid, predictedBlinks50m,'r')
    plt.plot(fid, actualBlinks50m,'b--')
    plt.ylim(-60, 20)

    #plt.figure()
    #plt.plot(fid, csdfilt, 'r')
    #plt.plot(fid, csdfiltbin,'b')
    
    #print(bStartFrame, bStopFrame, bDuration, bCount)
    #print(bmStartFrame, bmStopFrame, bmDuration, bmCount)
    #print(bmrStartFrame, bmrStopFrame, bmrDuration, bmrCount)


    
    
    
    
    """
    
    Av, framenumber = framebyframe(filePath) 
    Avfiltered, Abg = substractBackground(Av)
    # save_data(filePath)
    AvfilteredBinary = convertBinary(Avfiltered)
    bStartFrame, bEndFrame = detectBlinks(framenumber, AvfilteredBinary)
    bStartFrameMerged, bEndFrameMerged = blinkMerge(bStartFrame,bEndFrame)
    blinkMergeBinary = createBinary(framenumber, bStartFrameMerged, bEndFrameMerged)
    print(bStartFrameMerged, bEndFrameMerged)
    # countevents(arraysbinary, framenumber)
    Tp, Fp, Fn, GTstartblink, GTendblink = compareResults(emily, bStartFrameMerged)
    GTbinary = createGTbin(framenumber, GTstartblink, GTendblink)
    saveEvaluation(Tp, Fp, Fn)
    blinkDur = estimateDuration(bStartFrameMerged, bEndFrameMerged)
    blinkInterval = estimateAverageBlinkInterval(bStartFrameMerged, bEndFrameMerged)
    differentiation = differentiationAttempt(Avfiltered)
    
    xmin = 100
    xmax = 700
    # ymin = 0
    # ymax = 5
    
    plt.figure()
    plt.plot(framenumber, Av, 'k')
    plt.xlim([xmin, xmax])

    plt.figure()
    plt.plot(framenumber, Abg)
    plt.xlim([xmin, xmax])
    plt.figure()
    plt.plot(framenumber, Avfiltered)

    
    plt.figure()
    plt.plot(framenumber, (AvfilteredBinary))
    plt.xlim([xmin, xmax])
    # plt.ylim([ymin,ymax])
    
    plt.figure()
    plt.plot(framenumber, blinkMergeBinary)
    plt.xlim([xmin, xmax])  
    # plt.ylim([ymin,ymax])
    
    plt.figure()
    plt.plot(framenumber, GTbinary)
    plt.xlim([xmin, xmax])
    # plt.ylim([ymin,ymax])

    plt.figure()
    plt.plot(framenumber[:-1], differentiation)
    



    # #SAVING DATA AND PLOTTING           
            
    # stefbinary = plotdigital(Af)
    # plt.figure()
    # plt.plot(framenumber, Af, 'y')
    # plt.plot(framenumber, A, 'b')
    # # plt.plot(framenumber, Abg, 'g')
    # plt.plot(framenumber, stefbinary, 'r')
    # save_data(filePath)
    # s, m, n= countevents(stefbinary, framenumber)
    # plt.show()
    
    """