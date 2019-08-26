w = [0,1]
x = [5,4,3,2,1]
y = [2,5,6,8,10]

def hypothesis(w,xi):
    # make dot product later
    return w[0] + w[1]*xi

def getCost(w,x,y):
    m = len(y)
    z = 0
    for i in range (m):
        z += (hypothesis(w,x[i])-y[i])**2
    z = z/(2*m)
    return z

print (getCost (w,x,y))