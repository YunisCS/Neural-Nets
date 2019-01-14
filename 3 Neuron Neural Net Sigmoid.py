import math
import time

def sigmoid(x, y, bias, theta1, theta2):
    return 1 / (1 + math.exp(-(bias + theta1 * x + theta2 * y)))

def L():
    likelyhood = 0
    for element in trainingSet:
        sigma1 = sigmoid(element[0], element[1], bias1, theta1, theta3)
        sigma2 = sigmoid(element[0], element[1], bias2, theta2, theta4)
        sigma3 = sigmoid(sigma1, sigma2, bias3, theta5, theta6)
        likelyhood += element[2] * math.log(sigma3) + (1 - element[2]) * math.log(1 - sigma3 + 0.000000000000001)
    return likelyhood


trainingSet = []
'''
file = open("data.csv")
n = int(file.readline()[:-2])

for i in range(n):
    w, h, c = file.readline()[:-1].split()
    w, h, c = float(w), float(h), int(c[:1])
    trainingSet.append([w, h, c])
'''   
n = 10
 
for i in range(1,n + 1):
    w = math.pi * i / n * 0.7
    h = math.sin(w)
    c = 0
    trainingSet.append([w, h, c])
    h = h + .25
    c = 1
    trainingSet.append([w, h, c])
 
lineData = []

theta1 = 1
theta2 = 1
theta3 = 1
theta4 = 1
theta5 = 1
theta6 = 1
bias1 = 1
bias2 = 1
bias3 = 1
thetad = [0] * 9
alpha = 0.01
minD = []

for i in range(10 ** 5):
    
    for element in trainingSet:
        
        sigma1 = sigmoid(element[0], element[1], bias1, theta1, theta3)
        sigma2 = sigmoid(element[0], element[1], bias2, theta2, theta4)
        sigma3 = sigmoid(sigma1, sigma2, bias3, theta5, theta6)
        
        dLdsigma3 = (element[2] * 1 / (sigma3 + 0.00000000000000000000001)) + (1 - element[2]) * -1 / (1 - sigma3 + 0.0000000000000000001) 
        
        thetad[0] = dLdsigma3 * sigma3 * (1 - sigma3) * theta5 * sigma1 * (1 - sigma1) * element[0]
        thetad[1] = dLdsigma3 * sigma3 * (1 - sigma3) * theta6 * sigma2 * (1 - sigma2) * element[0]
        thetad[2] = dLdsigma3 * sigma3 * (1 - sigma3) * theta5 * sigma1 * (1 - sigma1) * element[1]
        thetad[3] = dLdsigma3 * sigma3 * (1 - sigma3) * theta6 * sigma2 * (1 - sigma2) * element[1]
        thetad[4] = dLdsigma3 * sigma3 * (1 - sigma3) * sigma1
        thetad[5] = dLdsigma3 * sigma3 * (1 - sigma3) * sigma2
    
        thetad[6] = dLdsigma3 * sigma3 * (1 - sigma3) * theta5 * sigma1 * (1 - sigma1)
        thetad[7] = dLdsigma3 * sigma3 * (1 - sigma3) * theta6 * sigma2 * (1 - sigma2)
        thetad[8] = dLdsigma3 * sigma3 * (1 - sigma3) 
        
        
        theta1 += thetad[0] * alpha
        theta2 += thetad[1] * alpha
        theta3 += thetad[2] * alpha
        theta4 += thetad[3] * alpha
        theta5 += thetad[4] * alpha
        theta6 += thetad[5] * alpha
        bias1 += thetad[6] * alpha
        bias2 += thetad[7] * alpha
        bias3 += thetad[8] * alpha
        
        
        
    maxD = 0
    for change in thetad:
        if maxD < abs(change):
            maxD = abs(change)
    minD.append(maxD)
#    if maxD < 10 ** -10:
#        print("Change is less than threshold")
#        break

#for element in trainingSet:
#    print(element[0], element[1], sigmoid(element[0], element[1], bias1, theta1, theta3),
#          sigmoid(element[0], element[1], bias2, theta2, theta4))

for element in trainingSet:
    print(round(element[0], 2), round(element[1], 2), element[2], ":", round(sigmoid(sigmoid(element[0], element[1], bias1, theta1, theta3),
                  sigmoid(element[0], element[1], bias2, theta2, theta4),
                  bias3, theta5, theta6), 2))

print("\n")
print("Likelyhood:", L(), "\n")
print("Theta1:", theta1)
print("Theta2:", theta2)
print("Theta3:", theta3)
print("Theta4:", theta4)
print("Theta5:", theta5)
print("Theta6:", theta6)
print("Bias1:", bias1)
print("Bias2:", bias2)
print("Bias3:", bias3)


'''

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

plotData = []
ax = plt.figure().add_subplot(111, axisbg='grey')
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


    
plt.plot(lineData)
plt.show()
'''