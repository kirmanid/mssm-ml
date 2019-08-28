w = [1,1]
x = [[1,5],[1,4],[1,3],[1,2],[1,1]]
y = [2,5,6,8,10]

learningRate = 0.1

def hypothesis(w,xRow):
    # make dot product later
    return w[0]*xRow[0] + w[1]*xRow[1]

def getCost(w,x,y):
    m = len(y)
    z = 0
    for i in range (m):
        z += (hypothesis(w,x[i])-y[i])**2
    z = z/(2*m)
    return z

def takeCostDeriv(w,x,y):
    m = len(y)
    d = [0]*len(w)
    for weightIndex in range (len(w)):
        z = 0
        for i in range (m):
            z += (hypothesis(w,x[i])-y[i])*x[i][weightIndex]
        z = z/m
        d[weightIndex] = z
    return d

def doGradientDescent(learningRate, w, x, y):
    d = takeCostDeriv(w,x,y)
    newWeights = [0] * len(w)
    for i in range (len(w)):
        newWeights[i] = w[i] - (learningRate*d[i])
    return newWeights

for i in range (200):
    w = doGradientDescent(learningRate, w, x, y)
    print (w)

print (getCost(w,x,y))