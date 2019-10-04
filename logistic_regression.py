import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white")

class LogisticRegression:
    def __init__(self,data,targetLabel):
        self.w = np.array([0]*(np.size(data[0]))) # sets the number of weights to the number of features +1
        self.targetLabel = targetLabel

    def hypothesis(self, xRow):
        h = np.dot (self.w, xRow)
        return (1/(1+np.e**(h*-1)))
    
    def predict(self, xRow):
        h = self.hypothesis (xRow)
        return (round(h))

    def getCost(self,x,y):
        m = len(y)
        z = 0
        for i in range (m):
            z += -1*y[i]*np.log(self.hypothesis(x[i])) - (1-y[i])*np.log(1-self.hypothesis(x[i]))
        z /= m
        return float(z)

    def takeCostDeriv(self,x,y):
        m = len(y)
        d = [0]*len(self.w)
        for weightIndex in range (len(self.w)):
            z = 0
            for i in range (m):
                z += (self.hypothesis(x[i])-y[i])*x[i][weightIndex]
            z = z/m
            d[weightIndex] = z
        return d

    def doGradientDescent(self, learningRate, x, y):
        d = self.takeCostDeriv(x,y)
        newWeights = np.zeros(len(self.w))
        for i in range (len(self.w)):
            newWeights[i] = self.w[i] - (learningRate*d[i])
        return newWeights

    def addXNoughts(self, x):
        x = np.reshape(x,((x.shape[0],np.size(x[0])))) # reshapes flat array or 2d array into transposed version
        x0 = np.ones((x.shape[0],1), dtype=np.uint8)
        x = np.hstack((x0,x))
        return x
    
    def featureSplit (self, data):
        dataCols = data.shape[1]
        x = data[:, 0:dataCols-1].astype(np.float32)
        x = self.addXNoughts(x)
        y = (data[:, dataCols-1:dataCols] == self.targetLabel).astype(int) # assuming label is last column
        return (x,y)

    def evalAccuracy(self, data):
        if (data.shape[0] == 0):
            return ("N/A ")
        x, y = self.featureSplit(data)
        correct = 0
        for i in range (y.shape[0]):
            if (self.predict(x[i]) == y[i] ):
                correct += 1
        correct /= y.shape[0]
        return correct
    
    def train(self, epochs, trainSet, learningRate,):
        x, y = self.featureSplit(trainSet)
        for i in range (epochs):
            self.w = self.doGradientDescent(learningRate, x, y)
            print ("weights (epoch " + str(i) + "): "+ str(self.w))
        print ("\nFinal Weights: " + str(self.w))
        print("\nFinal Cost (Training Set): " + str(self.getCost(x,y)))

learningRate = 0.1
epochs = 200
testSetRatio = 0.2

data = sns.load_dataset('iris').to_numpy()[:,2:5]
np.random.shuffle(data)

testTrainIndex = int(testSetRatio*data.shape[0])
testSet = data[:testTrainIndex]
trainSet = data[testTrainIndex:]

a = LogisticRegression(data, "setosa")
a.train(epochs, trainSet, learningRate)

xTest, yTest = a.featureSplit(testSet)
print ("\nFinal Cost (Test Set): " + str(a.getCost(xTest, yTest)))
print("\nAccuracy (Training Set): " + str(100*a.evalAccuracy(trainSet)) + "%")
print("\nAccuracy (Test Set): " + str(100*a.evalAccuracy(testSet)) + "%\n")

xx, yy = np.mgrid[4:8:0.1 , 2:4.5:0.1 ]
grid = np.c_[xx.ravel(), yy.ravel()]
grid = a.addXNoughts(grid)

# predict each element on grid, plot like on email