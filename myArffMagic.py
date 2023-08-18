import sys
import numpy as np
import os
import arff
import random
import math

test_place = 0 #The 10% that becomes the test set, 0 for first 10%, 1 for second 10% etc.
        
def getWindow(activity, x_raw, y_raw, z_raw):
    if len(np.unique(activity[:180])) == 1 and len(np.unique(activity[:200])) != 1:
        samples = 180
        while len(np.unique(activity[:samples])) == 1:
            samples += 1
        x, y, z = x_raw[:samples - 1], y_raw[:samples - 1], z_raw[:samples - 1]
        return x, y, z, activity, x_raw, y_raw, z_raw, samples-1
    elif len(np.unique(activity[:200])) != 1:
        while len(np.unique(activity[:200])) != 1:
            activity.pop(0)
            x_raw.pop(0)
            y_raw.pop(0)
            z_raw.pop(0)
    x, y, z = x_raw[:200], y_raw[:200], z_raw[:200]
    return x, y, z, activity, x_raw, y_raw, z_raw, 200

def getBinForValue(val):
    bindex = 0
    val = val / 2.5
    if val >= -1 and val <= 7:
        bindex = np.floor(val) + 2
    elif val > 7:
        bindex = 9
    return int(bindex)

def createBins(x, y, z):
    x_bins = np.histogram(x, bins=10, density=False)[0].astype(float).tolist()
    y_bins = np.histogram(y, bins=10, density=False)[0].astype(float).tolist()
    z_bins = np.histogram(z, bins=10, density=False)[0].astype(float).tolist()
    x_bin = [float(i/np.sum(x_bins)) for i in x_bins]
    y_bin = [float(i/np.sum(y_bins)) for i in y_bins]
    z_bin = [float(i/np.sum(z_bins)) for i in z_bins]
    return x_bin, y_bin, z_bin

def createAverages(x, y, z):
    return np.average(x), np.average(y), np.average(z)

def getPeaks(data):
    peaks = []
    if data[0] > data[1]:
        peaks.append(0)
    for i in range(1, len(data) - 2):
        if(data[i] > data[i-1] and data[i] > data[i+1]):
            peaks.append(i)
    if (data[-1] > data[-2]):
        peaks.append(len(data) - 1)
    return peaks

def calculateDistance(peaks):
    dists = []
    for i in range(len(peaks) - 2):
        dists.append(peaks[i+1] - peaks[i])
    
    sum = np.sum(dists)
    distance = (sum / len(dists)) * 10
    return distance

def calculatePeakDistance(x, y, z):
    x_peaks = getPeaks(x)
    x_distance = calculateDistance(x_peaks)
    y_peaks = getPeaks(y)
    y_distance = calculateDistance(y_peaks)
    z_peaks = getPeaks(z)
    z_distance = calculateDistance(z_peaks)
    return x_distance, y_distance, z_distance


def calculateAbsoluteDev(arr):
    dev = 0
    for i in arr:
        dev += np.abs(i - (np.average(arr)))
    dev = dev / len(arr)
    return dev


def getAbsoluteDev(x, y, z):
    x_abs = calculateAbsoluteDev(x)
    y_abs = calculateAbsoluteDev(y)
    z_abs = calculateAbsoluteDev(z)
    return x_abs, y_abs, z_abs

def calculateStandardDev(arr):
    sdev = 0
    for i in arr:
        sdev += pow((i - np.average(arr)), 2)
    sdev = np.sqrt(sdev) / len(arr)
    return sdev


def getStandardDev(x, y, z):
    x_sdev = calculateStandardDev(x)
    y_sdev = calculateStandardDev(y)
    z_sdev = calculateStandardDev(z)
    return x_sdev, y_sdev, z_sdev

def getResultant(x, y, z):
    sum = 0
    for i in range(len(x)):
        square = pow(x[i], 2) + pow(y[i], 2) + pow(z[i], 2)
        sum += np.sqrt(square)
    resultant = sum / len(x)
    return (resultant)

def getTrainTestSplit(arr, oneSecond):
    totalWindows = math.floor(len(arr)/oneSecond) - 9
    tenPercentWindows = math.floor((totalWindows - 18)/10)
    testStart = round(test_place * (totalWindows/10)) * oneSecond
    testEnd = round(testStart + (tenPercentWindows * oneSecond) + (9 * oneSecond))

    train1 = arr[:testStart]
    train2 = arr[testEnd:]
    test = arr[testStart:testEnd]
    return train1, train2, test


def makeArffFile(file, number):
    activity = []
    x_raw = []
    y_raw = []
    z_raw = []
    outdata = []
    outdata2 = []

    with open(file) as infile:
        for line in infile:
            line = line[:-2].split(',')
            activity.append(line[1])
            x_raw.append(float(line[3]))
            y_raw.append(float(line[4]))
            z_raw.append(float(line[5]))

        activities = sorted(list(set(activity)))
        #This code is for having evenly spaced random windows, while already creating train and test split to avoid any overlap in these sets later on.
        
        for act in activities:
            print(act, test_place)
            first = activity.index(act)
            last = len(activity) - 1 - activity[::-1].index(act)
            
            totalReadings = last - first
            newWindowStart = round(totalReadings/180)
            oneSecond = round(totalReadings/180)

            act_x_raw = x_raw[first+30*oneSecond:last+1]
            act_y_raw = y_raw[first+30*oneSecond:last+1]
            act_z_raw = z_raw[first+30*oneSecond:last+1]


            x_Train1, x_Train2, x_Test = getTrainTestSplit(act_x_raw, oneSecond)
            y_Train1, y_Train2, y_Test = getTrainTestSplit(act_y_raw, oneSecond)
            z_Train1, z_Train2, z_Test = getTrainTestSplit(act_z_raw, oneSecond)

            

            while len(x_Train1) >= 200 and len(y_Train1) >= 200 and len(z_Train1) >= 200:
                x_window = x_Train1[:200]
                y_window = y_Train1[:200]
                z_window = z_Train1[:200]
                x_bin, y_bin, z_bin = createBins(x_window, y_window, z_window)
                x_avg, y_avg, z_avg = createAverages(x_window, y_window, z_window)
                x_peak, y_peak, z_peak = calculatePeakDistance(x_window, y_window, z_window)
                x_abs, y_abs, z_abs = getAbsoluteDev(x_window, y_window, z_window)
                x_sdev, y_sdev, z_sdev = getStandardDev(x_window, y_window, z_window)
                resultant = getResultant(x_window, y_window, z_window)

                nonExisting = ['0'] *48

                data = []
                data.append(act)
                data += x_bin
                data += y_bin
                data += z_bin
                data.extend((x_avg, y_avg, z_avg, x_peak, y_peak, z_peak, x_abs, y_abs, z_abs, x_sdev, y_sdev, z_sdev))
                data += nonExisting
                data.extend(resultant, number)
                outdata.append(data)

                x_Train1 = x_Train1[newWindowStart:]
                y_Train1 = y_Train1[newWindowStart:]
                z_Train1 = z_Train1[newWindowStart:]

            while len(x_Train2) >= 200 and len(y_Train2) >= 200 and len(z_Train2) >= 200:
                x_window = x_Train2[:200]
                y_window = y_Train2[:200]
                z_window = z_Train2[:200]
                x_bin, y_bin, z_bin = createBins(x_window, y_window, z_window)
                x_avg, y_avg, z_avg = createAverages(x_window, y_window, z_window)
                x_peak, y_peak, z_peak = calculatePeakDistance(x_window, y_window, z_window)
                x_abs, y_abs, z_abs = getAbsoluteDev(x_window, y_window, z_window)
                x_sdev, y_sdev, z_sdev = getStandardDev(x_window, y_window, z_window)
                resultant = getResultant(x_window, y_window, z_window)

                nonExisting = ['0'] *48

                data = []
                data.append(act)
                data += x_bin
                data += y_bin
                data += z_bin
                data.extend((x_avg, y_avg, z_avg, x_peak, y_peak, z_peak, x_abs, y_abs, z_abs, x_sdev, y_sdev, z_sdev))
                data += nonExisting
                data.extend(resultant, number)
                outdata.append(data)

                x_Train2 = x_Train2[newWindowStart:]
                y_Train2 = y_Train2[newWindowStart:]
                z_Train2 = z_Train2[newWindowStart:]

            while len(x_Test) >= 200 and len(y_Test) >= 200 and len(z_Test) >= 200:
                x_window = x_Test[:200]
                y_window = y_Test[:200]
                z_window = z_Test[:200]
                x_bin, y_bin, z_bin = createBins(x_window, y_window, z_window)
                x_avg, y_avg, z_avg = createAverages(x_window, y_window, z_window)
                x_peak, y_peak, z_peak = calculatePeakDistance(x_window, y_window, z_window)
                x_abs, y_abs, z_abs = getAbsoluteDev(x_window, y_window, z_window)
                x_sdev, y_sdev, z_sdev = getStandardDev(x_window, y_window, z_window)
                resultant = getResultant(x_window, y_window, z_window)

                nonExisting = ['0'] *48

                data = []
                data.append(act)
                data += x_bin
                data += y_bin
                data += z_bin
                data.extend((x_avg, y_avg, z_avg, x_peak, y_peak, z_peak, x_abs, y_abs, z_abs, x_sdev, y_sdev, z_sdev))
                data += nonExisting
                data.extend(resultant, number)
                outdata2.append(data)

                x_Test = x_Test[newWindowStart:]
                y_Test = y_Test[newWindowStart:]
                z_Test = z_Test[newWindowStart:]
            
        
        arff.dump('%s_TrainSplit_split_%s.arff' % (file.split('.')[0], str(test_place)), outdata, relation='person_activities_labeled', names=['ACTIVITY', 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'XAVG', 'YAVG', 'ZAVG', 'XPEAK', 'YPEAK', 'ZPEAK', 'XABSOLDEV', 'YABSOLDEV', 'ZABSOLDEV', 'XSTANDDEV', 'YSTANDDEV', 'ZSTANDDEV', 'XVAR', 'YVAR', 'ZVAR', 'XMFCC0', 'XMFCC1', 'XMFCC2', 'XMFCC3', 'XMFCC4', 'XMFCC5', 'XMFCC6', 'XMFCC7', 'XMFCC8', 'XMFCC9', 'XMFCC10', 'XMFCC11', 'XMFCC12', 'YMFCC0', 'YMFCC1', 'YMFCC2', 'YMFCC3', 'YMFCC4', 'YMFCC5', 'YMFCC6', 'YMFCC7', 'YMFCC8', 'YMFCC9', 'YMFCC10', 'YMFCC11', 'YMFCC12', 'ZMFCC0', 'ZMFCC1', 'ZMFCC2', 'ZMFCC3', 'ZMFCC4', 'ZMFCC5', 'ZMFCC6', 'ZMFCC7', 'ZMFCC8', 'ZMFCC9', 'ZMFCC10', 'ZMFCC11', 'ZMFCC12', 'XYCOS', 'XZCOS', 'YZCOS', 'XYCOR', 'XZCOR', 'YZCOR', 'RESULTANT', 'class'])

        arff.dump('%s_TestSplit_split_%s.arff' % (file.split('.')[0], str(test_place)), outdata2, relation='person_activities_labeled', names=['ACTIVITY', 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'XAVG', 'YAVG', 'ZAVG', 'XPEAK', 'YPEAK', 'ZPEAK', 'XABSOLDEV', 'YABSOLDEV', 'ZABSOLDEV', 'XSTANDDEV', 'YSTANDDEV', 'ZSTANDDEV', 'XVAR', 'YVAR', 'ZVAR', 'XMFCC0', 'XMFCC1', 'XMFCC2', 'XMFCC3', 'XMFCC4', 'XMFCC5', 'XMFCC6', 'XMFCC7', 'XMFCC8', 'XMFCC9', 'XMFCC10', 'XMFCC11', 'XMFCC12', 'YMFCC0', 'YMFCC1', 'YMFCC2', 'YMFCC3', 'YMFCC4', 'YMFCC5', 'YMFCC6', 'YMFCC7', 'YMFCC8', 'YMFCC9', 'YMFCC10', 'YMFCC11', 'YMFCC12', 'ZMFCC0', 'ZMFCC1', 'ZMFCC2', 'ZMFCC3', 'ZMFCC4', 'ZMFCC5', 'ZMFCC6', 'ZMFCC7', 'ZMFCC8', 'ZMFCC9', 'ZMFCC10', 'ZMFCC11', 'ZMFCC12', 'XYCOS', 'XZCOS', 'YZCOS', 'XYCOR', 'XZCOR', 'YZCOR', 'RESULTANT', 'class'])


for i in range(0,10):
    test_place = i
    for file in os.listdir(os.path.join(sys.path[0], 'wisdm-dataset/raw/phone/gyro')):
        if not file.startswith('.') and not file.__contains__('Split') and file.__contains__('16'):
            print(file)
            makeArffFile(os.path.join(sys.path[0], 'wisdm-dataset/raw/phone/gyro/' + file), file.split('_')[1])