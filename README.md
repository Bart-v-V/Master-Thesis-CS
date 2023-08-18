# Master-Thesis-CS

The code used to do the experiments and get the results during the master thesis.

This repository contains: 
  - GyroRead4.apk, the application used during the thesis for recording the gyroscope data.
    This application records the data from all 3 axes at a suggested 20 Hz sampling rate,
    and can be set to record for a specific time or indefinitly.
  
  - Instructions.txt, the instructions on how to use the application.
    
  - collectedToRaw.py, used to create a file in the raw file format from the recorded data.
    
  - myArffMagic.py, creates the arff files from the raw files. This does all the calculations needed and creates 10 splits for 10-fold cross-validation.
    
  - ourResults.py, creates the identification models using the 10 splits and evaluates the performance of the models.
    This code can also be used to test on data with added noise.
