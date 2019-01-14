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
##file = open("trainingData.csv")
##n = int(file.readline()[:-2])
##
##for i in range(n):
##    w, h, c = file.readline()[:-1].split()
##    w, h, c = float(w), float(h), int(c[:1])
##    trainingSet.append([w, h, c])
##
##
##
##wght = []
##hght = []
##for element in trainingSet:
##    wght.append(element[0])
##    hght.append(element[1])
##  
###wAv = np.average(wght)
##wAv = 5.49
##wStd = 2.41
##hAv = 10.34
##hStd = 4.32
###hAv = np.average(hght)
###wStd = np.std(wght)
###hStd = np.std(hght)
##for element in trainingSet:
##    element[0] = (element[0] - wAv) / wStd
##    element[1] = (element[1] - wAv) / hStd
n = 10

for i in range(1,n):
    w = math.pi * i / n
    h = math.sin(w)
    c = 0
    trainingSet.append([w, h, c])
    h = h + .25
    c = 1
    trainingSet.append([w, h, c])
##print('the set',trainingSet)
##sys.exit
    
WI = np.array([[-.5,.5],[-.25,.75]])
BI = np.array([0.3,.20])
WH = np.array([[0.1,-0.2],[0.3,-0.4],[0.5,-0.6]])
BH = np.array([0.7,-0.8,0.9])
WO = np.array([-.66,.33,.33])
BO = 0


weights = {"i" : {"t" : [.3, -.5, .5], "b" : [0.2, -.25, .75]},
           "h" : {"t" : [0.7, .1, -.2], "m" : [-.8, 0.3, -.4], "b" : [0.9, .5, -.6]},
           "o" : {"t" : [0, -.66, .33, .33]}}

dweights = {"i" : {"t" : [0, 0, 0], "b" : [0, 0, 0]},
           "h" : {"t" : [0, 0, 0], "m" : [0, 0, 0], "b" : [0, 0, 0]},
           "o" : {"t" : [0, 0, 0, 0]}}

sums = {"i" : {"t" : 0, "b" : 0},
        "h" : {"t" : 0, "m" : 0, "b" : 0},
        "o" : {"t" : 0}}

alpha = 10 ** -2
beta = 10 ** -2

for i in range(10 ** 1):
    
    for element in trainingSet:
        XI = np.array(element[:-1])
        print('XI,WI,BI \n',XI,WI,BI)
        print('sumI \n',sumFunc("i", "t", element[:-1]))
        sigmaIT = sigmoid(sumFunc("i", "t", element[:-1]))
        sigmaIB = sigmoid(sumFunc("i", "b", element[:-1]))
        SumI = WI.dot(XI) + BI
        sigmaI = 1/(1+np.exp(-SumI))
        print('SumI,sigmaI \n',SumI,sigmaI)
        print('\n SumI.shape\n',SumI.shape)
        print('\n sigmaI.shape\n',sigmaI.shape)
        print('SigmaIT,SigmaIB \n',sigmaIT,sigmaIB)

        SumH = WH.dot(sigmaI) + BH
        print('SumH \n',SumH)
        reluH = np.maximum(0,SumH)
        print('reluH \n',reluH)
        reluHT = relu(sumFunc("h", "t", [sigmaIT, sigmaIB]))
        reluHM = relu(sumFunc("h", "m", [sigmaIT, sigmaIB]))
        reluHB = relu(sumFunc("h", "b", [sigmaIT, sigmaIB]))
        print('reluHTMB \n',reluHT, reluHM, reluHB)

        SumO = WO.dot(reluH) + BO
        sigmaOut = 1/(1+np.exp(-SumO))
        print('sigmaOut \n',sigmaOut)
        sigmaO = sigmoid(sumFunc("o", "t", [reluHT, reluHM, reluHB]))
        print('sigmaO \n',sigmaO)
        
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
        print('dweights o t \n', dweights["o"]["t"])

        dBO = dldsigmaO * sigmaOut * (1 - sigmaOut)
        dWO = dBO * reluH
        print('dBO \n',dBO,'\n dWO \n',dWO)
        
        
        dweights["h"]["t"][0] = dweights["o"]["t"][0] * weights["o"]["t"][1] * reluP(sums["h"]["t"])
        dweights["h"]["t"][1] = dweights["h"]["t"][0] * sigmaIT
        dweights["h"]["t"][2] = dweights["h"]["t"][0] * sigmaIB
        
        dweights["h"]["m"][0] = dweights["o"]["t"][0] * weights["o"]["t"][2] * reluP(sums["h"]["m"])
        dweights["h"]["m"][1] = dweights["h"]["m"][0] * sigmaIT
        dweights["h"]["m"][1] = dweights["h"]["m"][0] * sigmaIB

        
        dweights["h"]["b"][0] = dweights["o"]["t"][0] * weights["o"]["t"][3] * reluP(sums["h"]["b"])
        dweights["h"]["b"][1] = dweights["h"]["b"][0] * sigmaIT
        dweights["h"]["b"][2] = dweights["h"]["b"][0] * sigmaIB
        print('\n dweights h 0 \n',dweights["h"]["t"][0],dweights["h"]["m"][0],dweights["h"]["b"][0])
        
        print('WO \n',WO, '\n reluH \n',reluH,'\n SumH \n', SumH, '\n reluH/SumH \n',reluH/SumH)
        dBH = dBO*np.multiply(WO,reluH/SumH)
        dBH =dBH.reshape(3,1)
        print('dBH \n', dBH,'\n dBH.shape \n',dBH.shape)
        print('\n sigmaI sigmaI.shape\n',sigmaI,sigmaI.shape)
        dWH = sigmaI*dBH
        print(dWH)
        print('\n dweights h \n', dweights['h'])
        sys.exit()
        
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
    print('%.1f' %element[0],'%.1f' %element[1], '%.1f' %element[2], ":", '%.1f' %sigmaO)
print(weights)
    

