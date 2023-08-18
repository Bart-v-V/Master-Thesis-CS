import sys
import numpy as np

activity = 'D' #For the activity keys look at 'activity_key.txt'
testSubjectNumber = 1651

with open(sys.argv[1]) as infile, open('output.txt', 'w') as outfile:
    data = infile.readlines()
    data = data[0].split("/")

    x_data = np.array(data[0].split(',')).astype(np.float64)
    y_data = np.array(data[1].split(',')).astype(np.float64)
    z_data = np.array(data[2].split(',')).astype(np.float64)

    for i in range(len(x_data)):
        output = '%s,' % testSubjectNumber + activity + ',1,' + str(x_data[i]) + ',' + str(y_data[i]) + ',' + str(z_data[i]) + ';'
        outfile.write(output + '\n')
