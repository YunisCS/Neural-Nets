import math
import numpy as np
import sys
 
def sigmoid(dot):
    if dot < -20:
        return 0
    return 1 / (1 + (math.exp(-(dot))))
 
def sigmoidP(dot):
    return sigmoid(dot) * (1 - sigmoid(dot))
 
def relu(dot):
    return max(0, dot)
 
def reluP(dot):
    return relu(dot) / (dot+0.00000000000001)
 
def sumFunc(layer, level, x):
    Xsum = weights[layer][level][0]
    for i in range(1, len(x) + 1):
        Xsum += weights[layer][level][i] * x[i - 1]
    return Xsum
 
trainingSet = []

a = np.array([(1),
             (3),
             (5)])
    
b = np.array([(1, 2)])
print(b.shape(3, 1) * a)
'''
file = open("data.csv")
n = int(file.readline()[:-2])
 
for i in range(n):
    w, h, c = file.readline()[:-1].split()
    w, h, c = float(w), float(h), int(c[:1])
    trainingSet.append([w, h, c])
'''
n = 40
 
for i in range(1,n + 1):
    w = math.pi * i / n * 4
    h = math.sin(w)
    c = 0
    trainingSet.append([w, h, c])
    h = h + .25
    c = 1
    trainingSet.append([w, h, c])
 
'''
wght = []
hght = []
for element in trainingSet:
    wght.append(element[0])
    hght.append(element[1])
 
#wAv = np.average(wght)
wAv = 5.49
wStd = 2.41
hAv = 10.34
hStd = 4.32
#hAv = np.average(hght)
#wStd = np.std(wght)
#hStd = np.std(hght)
for element in trainingSet:
    element[0] = (element[0] - wAv) / wStd
    element[1] = (element[1] - wAv) / hStd
'''
 
weights = {"i" : {"t" : [0, .5, -.5], "b" : [0, -.5, .5]},
           "h" : {"t" : [0, .6, -.6], "m" : [0, -0.6, 0.6], "b" : [0, .4, -.4]},
           "o" : {"t" : [0, -.66, .33, .33]}}
 
dweights = {"i" : {"t" : [0, 0, 0], "b" : [0, 0, 0]},
           "h" : {"t" : [0, 0, 0], "m" : [0, 0, 0], "b" : [0, 0, 0]},
           "o" : {"t" : [0, 0, 0, 0]}}
 
sums = {"i" : {"t" : 0, "b" : 0},
        "h" : {"t" : 0, "m" : 0, "b" : 0},
        "o" : {"t" : 0}}
 
alpha = 10 ** -2
beta = 10 ** -2
 

for i in range(10 ** 5):
   
    for element in trainingSet:
       
        sigmaIT = sigmoid(sumFunc("i", "t", element[:-1]))
        sigmaIB = sigmoid(sumFunc("i", "b", element[:-1]))
       
        reluHT = relu(sumFunc("h", "t", [sigmaIT, sigmaIB]))
        reluHM = relu(sumFunc("h", "m", [sigmaIT, sigmaIB]))
        reluHB = relu(sumFunc("h", "b", [sigmaIT, sigmaIB]))
 
        sigmaO = sigmoid(sumFunc("o", "t", [reluHT, reluHM, reluHB]))
       
        for layer, values in sums.items():
            for level, values in sums[layer].items():
                if layer == "i":
                    sums[layer][level] = sumFunc(layer, level, element[:-1])
                elif layer == "h":
                    sums[layer][level] = sumFunc(layer, level, [sigmaIT, sigmaIB])
                else:
                    sums[layer][level] = sumFunc(layer, level, [reluHT, reluHM, reluHB])
       
        dldsigmaO = (element[2] * 1 / (sigmaO + 0.00000000000000000000001)) + (1 - element[2]) * -1 / (1 - sigmaO + 0.0000000000000000001)
        
        dweights["o"]["t"][0] = dldsigmaO * sigmaO * (1 - sigmaO)
        dweights["o"]["t"][1] = dweights["o"]["t"][0] * reluHT
        dweights["o"]["t"][2] = dweights["o"]["t"][0] * reluHM
        dweights["o"]["t"][3] = dweights["o"]["t"][0] * reluHB
       
        dweights["h"]["t"][0] = dweights["o"]["t"][0] * weights["o"]["t"][1] * reluP(sums["h"]["t"])
        dweights["h"]["t"][1] = dweights["h"]["t"][0] * sigmaIT
        dweights["h"]["t"][2] = dweights["h"]["t"][0] * sigmaIB
        
        dweights["h"]["m"][0] = dweights["o"]["t"][0] * weights["o"]["t"][2] * reluP(sums["h"]["m"])
        dweights["h"]["m"][1] = dweights["h"]["m"][0] * sigmaIT
        dweights["h"]["m"][1] = dweights["h"]["m"][0] * sigmaIB
       
        dweights["h"]["b"][0] = dweights["o"]["t"][0] * weights["o"]["t"][3] * reluP(sums["h"]["b"])
        dweights["h"]["b"][1] = dweights["h"]["b"][0] * sigmaIT
        dweights["h"]["b"][2] = dweights["h"]["b"][0] * sigmaIB
       
        dweights["i"]["t"][0] = (dweights["h"]["t"][0] * weights["h"]["t"][1] + dweights["h"]["b"][0] * weights["h"]["b"][1] + dweights["h"]["m"][0] * weights["h"]["m"][1]) * sigmoidP(sums["i"]["t"])
        dweights["i"]["t"][1] = dweights["i"]["t"][0] * element[0]
        dweights["i"]["t"][2] = dweights["i"]["t"][0] * element[1]
       
        dweights["i"]["b"][0] = (dweights["h"]["t"][0] * weights["h"]["t"][2] + dweights["h"]["b"][0] * weights["h"]["b"][2] + dweights["h"]["m"][0] * weights["h"]["m"][2]) * sigmoidP(sums["i"]["b"])
        dweights["i"]["b"][1] = dweights["i"]["b"][0] * element[0]
        dweights["i"]["b"][2] = dweights["i"]["b"][0] * element[1]
       
        
        weights["o"]["t"][0] += dweights["o"]["t"][0] * alpha
        weights["o"]["t"][1] += dweights["o"]["t"][1] * alpha
        weights["o"]["t"][2] += dweights["o"]["t"][2] * alpha
        weights["o"]["t"][3] += dweights["o"]["t"][3] * alpha
       
        weights["h"]["t"][0] += dweights["h"]["t"][0] * alpha
        weights["h"]["t"][1] += dweights["h"]["t"][1] * alpha
        weights["h"]["t"][2] += dweights["h"]["t"][2] * alpha
        
        weights["h"]["b"][0] += dweights["h"]["b"][0] * alpha
        weights["h"]["b"][1] += dweights["h"]["b"][1] * alpha
        weights["h"]["b"][2] += dweights["h"]["b"][2] * alpha
        
        weights["h"]["m"][0] += dweights["h"]["m"][0] * alpha
        weights["h"]["m"][1] += dweights["h"]["m"][1] * alpha
        weights["h"]["m"][2] += dweights["h"]["m"][2] * alpha
      
        weights["i"]["t"][0] += dweights["i"]["t"][0] * alpha
        weights["i"]["t"][1] += dweights["i"]["t"][1] * alpha
        weights["i"]["t"][2] += dweights["i"]["t"][2] * alpha
       
        weights["i"]["b"][0] += dweights["i"]["b"][0] * alpha
        weights["i"]["b"][1] += dweights["i"]["b"][1] * alpha
        weights["i"]["b"][2] += dweights["i"]["b"][2] * alpha
 
for element in trainingSet:
    sigmaIT = sigmoid(sumFunc("i", "t", element[:-1]))
    sigmaIB = sigmoid(sumFunc("i", "b", element[:-1]))
       
    reluHT = relu(sumFunc("h", "t", [sigmaIT, sigmaIB]))
    reluHM = relu(sumFunc("h", "m", [sigmaIT, sigmaIB]))
    reluHB = relu(sumFunc("h", "b", [sigmaIT, sigmaIB]))
        
    sigmaO = sigmoid(sumFunc("o", "t", [reluHT, reluHM, reluHB]))
    print(round(reluHT, 2), round(reluHM, 2), round(reluHB, 2), ":", round(element[0], 2), round(element[1], 2), round(element[2], 2), ":", round(sigmaO, 2))
print(weights)

import matplotlib.pyplot as mpl

