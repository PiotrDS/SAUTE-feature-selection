import numpy as np
from sklearn.neighbors import NearestNeighbors

def knnPredictProb(XTrain,XTest,y,knn=10):
    '''
    Implementing knnClasifier for partial labeling problem,
    Function returns index of label for each observation
    '''
    _,m = y.shape
    n,_ = XTest.shape
    
    kNN = NearestNeighbors(n_neighbors=knn)
    kNN.fit(XTrain)
    distances, indices = kNN.kneighbors(XTest)
    weights = np.arange(knn, 0, -1).reshape(knn, 1)

    scores = np.sum(y[indices] * weights, axis=1)
    
    
    probabilities = scores / np.sum(scores, axis=1, keepdims=True)

    returnArg = np.array([np.random.choice(a=m ,p=probabilities[i]) for i in range(n)])

    return returnArg

def knnPartialPredict(XTrain,XTest,y,knn=10, weightsType='dist'):
    '''
    Implementing knnClasifier for partial labeling problem,
    Function returns index of label for each observation
    '''
    
    kNN = NearestNeighbors(n_neighbors=knn)
    kNN.fit(XTrain)
    distances, indices = kNN.kneighbors(XTest)
    if weightsType == 'dist':
        weights = 1 - distances/np.sum(distances,axis=1)[:,None]
        Y = np.sum(y[indices]*weights[:,:,None], axis=1)
    elif weightsType == 'order':
        weights = np.arange(knn, 0, -1).reshape(knn, 1)
        Y = np.sum(y[indices]*weights, axis=1)

    maxIndex = np.argmax(Y, axis=1)

    return maxIndex

def knnPartial(X,y,knn=10, firstNeighbour=True):
    '''
    Implementing knnClasifier for partial labeling problem,
    Function returns index of label for each observation
    '''
    
    kNN = NearestNeighbors(n_neighbors=knn)
    kNN.fit(X)
    if firstNeighbour:
        distances, indices = kNN.kneighbors(X)
    else:
        distances, indices = kNN.kneighbors()
    
    weights = np.arange(knn, 0, -1).reshape(knn, 1)
    Y = np.sum(y[indices]*weights, axis=1)
    Y = Y*y
    
    maxIndex = np.argmax(Y, axis=1)

    return maxIndex, Y