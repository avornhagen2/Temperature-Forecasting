import sys

import numpy
import numpy as np
import csv

from models import createCLSTM, createLSTM
from utils import create_datasetLSTM
from matplotlib import pyplot as plt

def fileLen(fileName):
    with open(fileName) as f:
        return sum(1 for line in f)


def unscale(predMat, scaleMat):
    idx = 0
    for pred in predMat:
        predMat[idx][0] = pred[0] * scaleMat[1] + scaleMat[0]
        predMat[idx][1] = pred[1] * scaleMat[3] + scaleMat[2]
        # predMat[idx][2] = pred[2] * scaleMat[5] + scaleMat[4]  # Added this for 3 inputs
        # predMat[idx][3] = pred[3] * scaleMat[7] + scaleMat[6]  # Added this for 4 inputs
        idx += 1
    return predMat


def stripDate(dateIn):
    try:
        month, day, year = dateIn.split('/')
        return day, month
    except ValueError:
        print(dateIn)





numLines = fileLen('weatherFull.csv') - 1  # Remove Header Line
with open('weatherFull.csv') as file:
    weatherArr = np.zeros([numLines - 1, 2])

    weatherCSV = csv.reader(file)
    line = next(weatherCSV)
    idx = 0
    tempMaxTotal = 0
    tempMinTotal = 0
    while idx < numLines - 1:
        line = next(weatherCSV)

        # day, month = stripDate(line[1])
        tempMax = line[2]
        tempMin = line[3]
        seaLevelPressure = line[19]
        humidity = line[9]

        weatherArr[idx] = [float(tempMax), float(tempMin)]
        # weatherArr[idx] = [float(tempMax), float(tempMin), float(seaLevelPressure)]
        # weatherArr[idx] = [float(tempMax), float(tempMin), float(seaLevelPressure), float(humidity)]

        idx += 1


# print(weatherArr)

# Standard Normalization
maxMean = np.average(weatherArr[:, 0])
maxSTD = np.std(weatherArr[:, 0])
weatherArr[:, 0] = (weatherArr[:, 0] - maxMean) / maxSTD

minMean = np.average(weatherArr[:, 1])
minSTD = np.std(weatherArr[:, 1])
weatherArr[:, 1] = (weatherArr[:, 1] - minMean) / minSTD

# # added this for 3 inputs
# pressureMean = np.average(weatherArr[:, 2])
# pressureSTD = np.std(weatherArr[:, 2])
# weatherArr[:, 2] = (weatherArr[:, 2] - pressureMean) / pressureSTD
#
# # added this for 4 inputs
# humidityMean = np.average(weatherArr[:, 3])
# humiditySTD = np.std(weatherArr[:, 3])
# weatherArr[:, 3] = (weatherArr[:, 3] - humidityMean) / humiditySTD

scaleArr = [maxMean, maxSTD, minMean, minSTD]
stringArr = str(maxMean) + str(maxSTD) + str(minMean) + str(minSTD)

# scaleArr = [maxMean, maxSTD, minMean, minSTD, pressureMean, pressureSTD]
# stringArr = str(maxMean) + str(maxSTD) + str(minMean) + str(minSTD) + str(pressureMean) + str(pressureSTD)

# scaleArr = [maxMean, maxSTD, minMean, minSTD, pressureMean, pressureSTD, humidityMean, humiditySTD]
# stringArr = str(maxMean) + str(maxSTD) + str(minMean) + str(minSTD) + str(pressureMean) + str(pressureSTD) + str(humidityMean) + str(humiditySTD)

with open('scaleMat.txt', 'w') as file:
    file.write("maxMean maxSTD minMean minSTD\n")
    file.write(stringArr)
trainX, trainY, testX, testY = create_datasetLSTM(weatherArr, 10)

# clstmModel = createCLSTM(filters=64, denseNodes=[50, 2], num_dense_layers=2)


lstm = createLSTM()
history = lstm.fit(trainX, trainY, validation_split=.2, epochs=25, batch_size=32)

# clstm = createCLSTM()
# history = clstm.fit(trainX, trainY, validation_split=.2, epochs=25, batch_size=32)

plt.figure(0)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['MSE loss','Validation loss'], loc='upper left')
plt.savefig("loss.png")
plt.show()


predY = lstm.predict(testX)
# predY = clstm.predict(testX)
unscaledPred = unscale(predY, scaleMat=scaleArr)
unscaledTest = unscale(testY, scaleMat=scaleArr)

# plt.figure(1)
# plt.plot(unscaledTest[:, 1])
# plt.plot(unscaledPred[:, 1])
# plt.title('Minimum Temperatures')
# plt.legend(['True', 'Pred'], loc='upper left')
# plt.ylabel('Min Temp')
# plt.xlabel('Day')
# plt.savefig("mintemp.png")
# plt.show()
#
#
# plt.figure(2)
# plt.plot(unscaledTest[:, 0])
# plt.plot(unscaledPred[:, 0])
# plt.title('Maximum Temperatures')
# plt.legend(['True', 'Pred'], loc='upper left')
# plt.ylabel('Max Temp')
# plt.xlabel('Day')
# plt.savefig("maxtemp.png")
# plt.show()
#
# plt.figure(3)
# plt.plot(np.abs((unscaledTest[:, 1] - unscaledPred[:, 1])/unscaledTest[:, 1]) * 100)
#
# plt.title('Minimum Temperature Accuracy')
# plt.ylabel('Min Temp')
# plt.xlabel('Day')
# plt.savefig("minAcc.png")
# plt.show()
#
#
# plt.figure(4)
# plt.plot(np.abs((unscaledTest[:, 0] - unscaledPred[:, 0])/unscaledTest[:, 0]) * 100)
#
# plt.title('Maximum Temperature Accuracy')
# plt.ylabel('Accuracy Percentage')
# plt.xlabel('Day')
# plt.savefig("maxAcc.png")
# plt.show()
#
#
#
# plt.figure(5)
# plt.plot(np.abs((unscaledTest[:, 1] - unscaledPred[:, 1])))
# plt.title('Minimum Temperature Difference')
# plt.ylabel('Difference (degrees)')
# plt.xlabel('Day')
# plt.savefig("minDiff.png")
# plt.show()
#
# plt.figure(6)
# plt.plot(np.abs((unscaledTest[:, 0] - unscaledPred[:, 0])))
#
# plt.title('Maximum Temperature Difference')
# plt.ylabel('Difference (degrees)')
# plt.xlabel('Day')
# plt.savefig("maxDiff.png")
# plt.show()
#
# print("Average Humidity Difference = %.2f" % (np.average(np.absolute(unscaledTest[:, 3] - unscaledPred[:, 3]))))
# print("Average Sea Level Pressure Difference = %.2f" % (np.average(np.absolute(unscaledTest[:, 2] - unscaledPred[:, 2]))))
print("Average Min Difference = %.2f" % (np.average(np.absolute(unscaledTest[:, 1] - unscaledPred[:, 1]))))
print("Average Max Difference = %.2f" % (np.average(np.absolute(unscaledTest[:, 0] - unscaledPred[:, 0]))))

# print(unscaledTest[:, 1])
# print("----------------")
# print(unscaledTest[:, 0])
# print("----------------")
# print(unscaledTest[:, 2])
# print("----------------")
# print(unscaledTest[:, 3])


