import math

def sigmoid(x, tmpAlpha, tmpMu):
    return 1 / (1 + math.exp(-tmpAlpha * (x - tmpMu)))

def error(alpha, mu):
    errors = 0
    for tuple in trainingData:
        errors += (sigmoid(tuple[0], alpha, mu) - tuple[1]) ** 2
    return errors
    
n = int(input())
trainingData = []
for i in range(n):
    w, c = input().split()
    w, c = float(w), int(c)
    trainingData.append((w, c))

alpha = 2
mu = 4
#alphaIncrement = 0.1
muIncrement = 0.1
for i in range(100):
    errorVal = error(alpha, mu)
    
    checkUp = error(alpha, mu + muIncrement)
    checkDown = error(alpha, mu - muIncrement)
    if errorVal > checkUp and checkUp <= checkDown:
        mu += muIncrement
    elif errorVal > checkDown and checkDown < checkUp:
        mu -= muIncrement
    else:
        muIncrement += 0.1
        
#    checkUp = error(alpha + alphaIncrement, mu)
#    checkDown = error(alpha - alphaIncrement, mu)
#    if errorVal > checkUp and checkUp <= checkDown:
#        alpha += alphaIncrement
#    elif errorVal > checkDown and checkDown < checkUp:
#        alpha -= alphaIncrement
#    else:
#        alphaIncrement += 0.1
print(round(mu, 4))


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

plotData = []
mpl.rcParams.update({'font.size': 22})
trainingPlotData = np.array(trainingData)
x, y = trainingPlotData.T
plt.scatter(x, y)
xAxis = np.arange(0., 10., 0.1)
for i in range(100):
    plotData.append(sigmoid(i / 10, alpha, mu))
plt.plot(xAxis, plotData)
plt.show()


#