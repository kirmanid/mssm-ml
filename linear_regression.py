import numpy as np
import seaborn as sns

class LinearRegression:
    def __init__(self,x):
        self.w = np.array([1]*(np.size(x[0])+1)) # this sets the number of weights to the number of features +1

    def hypothesis(self, xRow):
        return np.dot (self.w, xRow)

    def getCost(self,x,y):
        m = len(y)
        z = 0
        for i in range (m):
            z += (self.hypothesis(x[i])-y[i])**2
        z = z/(2*m)
        return z

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
    
    def train(self, epochs, x, y, learningRate):
        x = self.addXNoughts(x)
        for i in range (epochs):
            self.w = self.doGradientDescent(learningRate, x, y)
            print ("epoch " + str(i))
            #print(self.getCost(x,y)) # shows progress
        print("\nFinal Cost:")
        print (self.getCost(x,y))

learningRate = 0.01
epochs = 200

iris = sns.load_dataset('iris').to_numpy()[:,:4].astype(np.float32)
np.random.shuffle(iris)

x = iris[:, 0:3]
y = iris[:, 3:4]


a = LinearRegression(x)
a.train(epochs,x,y,learningRate)

