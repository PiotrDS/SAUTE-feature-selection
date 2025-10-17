import numpy as np
import cvxpy as cp
from sklearn.neighbors import NearestNeighbors

def ipal(X,y_pl, knn=10, alpha=0.95, iterNo=20):

    '''
    Function ipal implements the IPAL algorithm to resolve ambiguity in partial labels.
    '''

    observationNo = X.shape[0]

    # Identify nearest neighbours for each observation
    kNN = NearestNeighbors(n_neighbors=knn)
    kNN.fit(X)
    distances, indices = kNN.kneighbors()


    # initialize weight matrix
    W = [[0 for _ in range(observationNo)] for __ in range(observationNo)]

    for observationIdx, idx in enumerate(indices):

        weightsNew = solveQuadr(observation=X[observationIdx, :], neighborsMatrix=X[idx, :])

        for j, i in enumerate(idx):
            W[i][observationIdx] = weightsNew[j,0]
    W = np.array(W)
    
    # update W matrix

    invD = np.diag(1/np.sum(W, axis=0))

    H = W @ invD

    # create labeling confidence matrix
    yCount = np.sum(y_pl, axis=1, keepdims=True)
    P = np.divide(y_pl,yCount) 

    F = P

    for i in range(iterNo):

        F = alpha * H.T @ F + (1-alpha)*P

        #rescale F matrix
        F =F * y_pl
        F = F / np.sum(F, axis=1, keepdims=True)

    Yprobs = np.sum(P, axis=0, keepdims=True) / np.sum(F, axis=0, keepdims=True) * F

    Y = np.argmax(Yprobs, axis=1)

    return Y, kNN, F


def predictIpal(X,X_new, y_pl, nn,alpha, iterNo=15):

    '''
    Function predictIpals implements the IPAL algorithm to classify the set X_new using PL-KNN algorithm trained on X and y_pl..
    '''

    YIpal, kNN, _ = ipal(X, y_pl, nn, alpha, iterNo)


    _, indices = kNN.kneighbors(X_new)

    observationNo = X_new.shape[0]    

    yPred = [None for _ in range(observationNo)]

    for observationIdx, idx in enumerate(indices):
        weights = solveQuadr(observation=X_new[observationIdx, :], neighborsMatrix=X[idx, :])

        labelsScores = {label: X_new[observationIdx].copy() for label in np.unique(YIpal[idx])}
        for (i,id) in enumerate(idx):
            sumLabels = weights[i,:] * X[id, :]
            labelsScores[YIpal[id]] -= sumLabels
        for label,score in labelsScores.items():
            labelsScores[label] = np.sum(score*score)
        predLabel = min(labelsScores, key=labelsScores.get) 

        yPred[observationIdx] = predLabel

    return np.array(yPred), YIpal


def solveQuadr(observation, neighborsMatrix):

    '''
    Function solveQuadr implements a solution for the quadratic-linear problem presented in the IPAL algorithm.
    '''

    N, d = neighborsMatrix.shape 

    P = neighborsMatrix @ neighborsMatrix.T  
    q = (-neighborsMatrix @ observation).reshape(-1, 1)
    G = -np.identity(N)
    h = np.zeros((N, 1))
    
    weights = cp.Variable((N, 1)) 

    objective = cp.Minimize(cp.quad_form(weights, P) + 2 * q.T @ weights)

    constraints = [G @ weights <= h]

    problem = cp.Problem(objective, constraints)

    result = problem.solve(solver=cp.OSQP)

    return weights.value  