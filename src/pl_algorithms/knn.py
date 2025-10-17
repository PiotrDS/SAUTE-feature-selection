import numpy as np
from sklearn.neighbors import NearestNeighbors

def knnPartialPredict(X,X_new,y,knn=10, weightsType='dist'):

    '''
    KNN - Classifier for partial labeling problems.

    This function implements a K-Nearest Neighbors classifier adapted for partial label learning.
    Makes a classification for the set X_new using PL-KNN algorithm trained on X and y_pl.
    It returns the index of the label for each observation.
    '''
    
    kNN = NearestNeighbors(n_neighbors=knn)
    kNN.fit(X)
    distances, indices = kNN.kneighbors(X_new)
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
    KNNClassifier for partial labeling problems.

    This class implements a K-Nearest Neighbors classifier adapted for partial label learning.
    Makes a classification for the set X using PL-KNN algorithm trained on X and y_pl.
    It returns the index of the label for each observation.
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