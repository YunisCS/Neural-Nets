import math

def sigmoid(x, y, theta0, theta1, theta2):
    return 1 / (1 + math.exp(-(theta0 + theta1 * x + theta2 * y)))

def L(theta0, theta1, theta2):
    likelyhood = 0
    for element in trainingSet:
        sig = sigmoid(element[0], element[1], theta0, theta1, theta2)
        likelyhood += element[2] * math.log(sig) + (1 - element[2]) * math.log(1 - sig)
    return likelyhood

trainingSet = []
file = open("data.csv")
n = int(file.readline()[:-2])

for i in range(n):
    w, h, c = file.readline()[:-1].split()
    w, h, c = float(w), float(h), int(c[:1])
    trainingSet.append([w, h, c])



theta0 = 1
theta1 = 1
theta2 = 1

alpha = 0.01

for i in range(10 ** 4):
    
    for element in trainingSet:
        theta0 += alpha * (element[2] - sigmoid(element[0], element[1], theta0, theta1, theta2))
        theta1 += alpha * (element[2] - sigmoid(element[0], element[1], theta0, theta1, theta2)) * element[0]
        theta2 += alpha * (element[2] - sigmoid(element[0], element[1], theta0, theta1, theta2)) * element[1]
        
        
print(theta0, theta1, theta2)

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

plotData = []
ax = plt.figure().add_subplot(111, axisbg='grey')
ax.set_xlabel("Weight")
ax.set_ylabel("Height")
mpl.rcParams.update({'font.size': 22})
ax.plot()
trainingPlotData = np.array(trainingSet)
x = []
y = []
z = []
for i in range(n):
    x.append(trainingSet[i][0])
    y.append(trainingSet[i][1])
    z.append(trainingSet[i][2])
plt.scatter(x, y, c = z, s = 100)

lineData = []
for i in range(11):
    lineData.append(theta0 / -theta2 + theta1 * i / -theta2)
plt.plot(lineData)
plt.show()


