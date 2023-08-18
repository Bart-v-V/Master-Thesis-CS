from sklearn.ensemble import RandomForestClassifier
import arff
import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
from statistics import mean
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Reading the initial model
with open(os.path.join(sys.path[0], 'FirstModel.pkl') ,'rb') as f:
    clf2 = pickle.load(f)

# Creating the arffdata array which will hold the unprocessed arff data
arffdata = []
noisedata = []
class_names = []

fullResults = []

# Results for any added test subjects
results1652 = []
results1653 = []
results1654 = []
resultsCombined = []

activityLabel = 'E' #The activity to build the model for. For the activity keys look at 'activity_key.txt'

# Collecting the data from the arff files, using try because number 14 does not exist everywhere
def createModel(split):
    for i in range (0,60):
        try:
            for row in arff.load(os.path.join(sys.path[0] ,'wisdm-dataset/raw/phone/gyro/data_16%s_gyro_phone_TrainSplit_final_split_%s_magic.arff' % (str(i).zfill(2), str(split)))):
                if not '16%s' % str(i).zfill(2) in class_names:
                    class_names.append('16%s' % str(i).zfill(2))
                if row[0]  == activityLabel:
                    arffdata.append(list(row))
        except:
            continue

# This is to load any altered files for specific test subject. For example if some have added noise and others do not. 59 does not exist so this number just gets skipped below.
    for i in range (0,60):
        try:
            if i == 59:
                for row in arff.load(os.path.join(sys.path[0] ,'wisdm-dataset/raw/phone/gyro/data_16%s_gyro_phone_noise_TestSplit_final_split_%s_magic.arff' % (str(i).zfill(2), str(split)))):
                    if row[0]  == activityLabel:
                        noisedata.append(list(row))
            elif i == 59:
                for row in arff.load(os.path.join(sys.path[0] ,'wisdm-dataset/raw/phone/gyro/data_16%s_gyro_phone_noise_TestSplit_final_split_%s_magic.arff' % (str(i).zfill(2), str(split)))):
                    if row[0]  == activityLabel:
                        noisedata.append(list(row))
            else: 
                for row in arff.load(os.path.join(sys.path[0] ,'wisdm-dataset/raw/phone/gyro/data_16%s_gyro_phone_TestSplit_final_split_%s_magic.arff' % (str(i).zfill(2), str(split)))):
                    if row[0]  == activityLabel:
                        noisedata.append(list(row))
        except:
            continue


    # Converting the arff data to an np array so we can create the datafram
    data = np.array(arffdata)
    noise = np.array(noisedata)

    # Collecting the training data we need into a dataframe
    df = pd.DataFrame({
        'Activity':data[:,0],
        'X0':data[:,1],
        'X1':data[:,2],
        'X2':data[:,3],
        'X3':data[:,4],
        'X4':data[:,5],
        'X5':data[:,6],
        'X6':data[:,7],
        'X7':data[:,8],
        'X8':data[:,9],
        'X9':data[:,10],
        'Y0':data[:,11],
        'Y1':data[:,12],
        'Y2':data[:,13],
        'Y3':data[:,14],
        'Y4':data[:,15],
        'Y5':data[:,16],
        'Y6':data[:,17],
        'Y7':data[:,18],
        'Y8':data[:,19],
        'Y9':data[:,20],
        'Z0':data[:,21],
        'Z1':data[:,22],
        'Z2':data[:,23],
        'Z3':data[:,24],
        'Z4':data[:,25],
        'Z5':data[:,26],
        'Z6':data[:,27],
        'Z7':data[:,28],
        'Z8':data[:,29],
        'Z9':data[:,30],
        'XAVG':data[:,31],
        'YAVG':data[:,32],
        'ZAVG':data[:,33],
        'XPEAK':data[:,34],
        'YPEAK':data[:,35],
        'ZPEAK':data[:,36],
        'XABSOLDEV':data[:,37],
        'YABSOLDEV':data[:,38],
        'ZABSOLDEV':data[:,39],
        'XSTANDDEV':data[:,40],
        'YSTANDDEV':data[:,41],
        'ZSTANDDEV':data[:,42],
        'RESULTANT':data[:,91],
        'class':data[:,92]
    })

    # Creating the feature and label 'identifiers'?
    x_train = df[['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'XAVG', 'YAVG', 'ZAVG', 'XPEAK', 'YPEAK', 'ZPEAK', 'XABSOLDEV', 'YABSOLDEV', 'ZABSOLDEV', 'XSTANDDEV', 'YSTANDDEV', 'ZSTANDDEV', 'RESULTANT']]
    y_train = df['class']

    #Creating the second dataframe for test data
    df2 = pd.DataFrame({
        'Activity':noise[:,0],
        'X0':noise[:,1],
        'X1':noise[:,2],
        'X2':noise[:,3],
        'X3':noise[:,4],
        'X4':noise[:,5],
        'X5':noise[:,6],
        'X6':noise[:,7],
        'X7':noise[:,8],
        'X8':noise[:,9],
        'X9':noise[:,10],
        'Y0':noise[:,11],
        'Y1':noise[:,12],
        'Y2':noise[:,13],
        'Y3':noise[:,14],
        'Y4':noise[:,15],
        'Y5':noise[:,16],
        'Y6':noise[:,17],
        'Y7':noise[:,18],
        'Y8':noise[:,19],
        'Y9':noise[:,20],
        'Z0':noise[:,21],
        'Z1':noise[:,22],
        'Z2':noise[:,23],
        'Z3':noise[:,24],
        'Z4':noise[:,25],
        'Z5':noise[:,26],
        'Z6':noise[:,27],
        'Z7':noise[:,28],
        'Z8':noise[:,29],
        'Z9':noise[:,30],
        'XAVG':noise[:,31],
        'YAVG':noise[:,32],
        'ZAVG':noise[:,33],
        'XPEAK':noise[:,34],
        'YPEAK':noise[:,35],
        'ZPEAK':noise[:,36],
        'XABSOLDEV':noise[:,37],
        'YABSOLDEV':noise[:,38],
        'ZABSOLDEV':noise[:,39],
        'XSTANDDEV':noise[:,40],
        'YSTANDDEV':noise[:,41],
        'ZSTANDDEV':noise[:,42],
        'RESULTANT':noise[:,91],
        'class':noise[:,92]
    })

    x_test = df2[['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'XAVG', 'YAVG', 'ZAVG', 'XPEAK', 'YPEAK', 'ZPEAK', 'XABSOLDEV', 'YABSOLDEV', 'ZABSOLDEV', 'XSTANDDEV', 'YSTANDDEV', 'ZSTANDDEV', 'RESULTANT']]
    y_test = df2['class']


    lst_accu_stratified = []

    predicted = np.array([])
    actual = np.array([])

    clf = RandomForestClassifier(n_estimators=10, random_state=42, max_features='sqrt', oob_score=True, warm_start=True)

    #arrays for the Out-of-Bag (OOB) scores and errors.
    oob_errors = []
    oob_scores = []

    #The start point of the random forest
    min_estimators = 10
    #The total number of trees used for the random forest
    max_estimators = 150


    #Creating the random forest in steps, to get OOB scores and errors
    for i in range(min_estimators, max_estimators + 1, 10):
        clf.set_params(n_estimators=i)
        clf.fit(x_train ,y_train)
        oob_errors.append(1-clf.oob_score_)
        oob_scores.append(clf.oob_score_)
        lst_accu_stratified.append(clf.score(x_test, y_test))


    y_pred_test = clf.predict(x_test)
    predicted = np.append(predicted, y_pred_test)
    actual = np.append(actual, y_test)
    print('oob_errors: ', oob_errors)


    # Saving the created model
    with open(os.path.join(sys.path[0], 'model.pkl') ,'wb') as f:
        pickle.dump(clf, f)

    #Code to get the accuracy results for specific test subjects, and those combined
    # right = [0] * 3
    # wrong = [0] * 3

    # for i in range(len(actual)):
    #     if actual[i] == '1652':
    #         if predicted[i] == '1652':
    #             right[0] += 1
    #         else:
    #             wrong[0] += 1
    #     if actual[i] == '1653':
    #         if predicted[i] == '1653':
    #             right[1] += 1
    #         else:
    #             wrong[1] += 1
    #     if actual[i] == '1654':
    #         if predicted[i] == '1654':
    #             right[2] += 1
    #         else:
    #             wrong[2] += 1

    # results1652.append(float(right[0])/float(right[0] + wrong[0]))
    # results1653.append(float(right[1])/float(right[1] + wrong[1]))
    # #resultsCombined.append(float(right[0] + right[1])/float(right[0] + wrong[0] + right[1] + wrong[1]))

    # results1654.append(float(right[2])/float(right[2] + wrong[2]))
    # resultsCombined.append(float(right[0] + right[1] + right[2])/float(right[0] + wrong[0] + right[1] + wrong[1] + right[2] + wrong[2]))

    # print('Accuracy 1652: ' + str(float(right[0])/float(right[0] + wrong[0])))
    # print('Accuracy 1653: ' + str(float(right[1])/float(right[1] + wrong[1])))
    # # print('Accuracy 1654: ' + str(float(right[2])/float(right[2] + wrong[2])))
    # # print('Accuracy Combined: ' + str(float(right[0] + right[1] + right[2])/float(right[0] + wrong[0] + right[1] + wrong[1] + right[2] + wrong[2])))

    # print('Accuracy Combined: ' + str(float(right[0] + right[1])/float(right[0] + wrong[0] + right[1] + wrong[1])))

    fullResults.append(clf.score(x_test, y_test))
    print(round(clf.score(x_test, y_test)*100,1))
    #print('oob-score: ', oob_scores[-1])



for j in range(0, 10):
    print(j)
    arffdata = []
    noisedata = []
    class_names = []
    createModel(j)

#The final results, both per fold and the mean of them
print(fullResults)
print("overall results: ", round(mean(fullResults)*100,1))

# print(results1652)
# print("1652 Accuracy: ", round(mean(results1652)*100,1))

# print(results1653)
# print("1653 Accuracy: ", round(mean(results1653)*100,1))

# print(results1654)
# print("1654 Accuracy: ", round(mean(results1654)*100,1))

# print(resultsCombined)
# print("Combined Accuracy: ", round(mean(resultsCombined)*100,1))
