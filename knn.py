import numpy as np
class Knn:
    def __init__(self, data, labels, k = 3, Testing = False):
        self.data = data
        self.labels = labels
        self.k = k
        self.i = 0
        self.testing = Testing
    
    def SetK(self, k):
        self.k = k
        self.i = 0

    def FindDistance(self, p1, p2):
        return np.sqrt(np.sum(np.power(p1.flatten() - p2.flatten(),2)))

    def TakeFirst(self, a):
        return a[0]

    def Predict(self, point):
        dist = []
        for (d, l) in zip(self.data, self.labels):
            dist.append((self.FindDistance(d, point), d, l))
        
        dist.sort(key=self.TakeFirst)

        neighbors = [dist[i][-1] for i in range(self.k)]
        
        if self.testing:
            print(self.i)
            self.i += 1

        return max(neighbors, key=neighbors.count)