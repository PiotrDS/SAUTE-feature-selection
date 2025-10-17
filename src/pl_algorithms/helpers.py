import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mutual_info_score
import torch

device = "gpu" if torch.cuda.is_available() else "cuda"

def discretizeData(data):

    '''
    Discretize data using formula (11) from saute article.
    '''

    numberOfObservation = data.shape[0]

    mu = np.mean(data, axis=0)
    mu = np.repeat(mu[np.newaxis,:],numberOfObservation,axis=0)

    std = np.std(data, axis=0)
    std = np.repeat(std[np.newaxis,:],numberOfObservation,axis=0)

    firstThreshold = mu - 2*std
    secondThreshold = mu - std
    thirdThreshold = mu + std
    fourthThreshold = mu + 2*std

    discretizedData = -2*(data <= firstThreshold) + (-1)*((firstThreshold<data)&(data<=secondThreshold)) + ((thirdThreshold<data) & (data<=fourthThreshold)) + 2*(fourthThreshold<data)

    return discretizedData

def calculateMutualInformation(data):
    '''
    Calculates mutural_info_score from sklearn package
    for every pair of variables from data
    '''
    # calculate number of features
    featureNumber = data.shape[1]
    # create dictionary for keeping mutal information beetwen variables
    MI = [[None for i in range(featureNumber)] for j in range(featureNumber)]
    # discretize data
    discretizedData = discretizeData(data)

    for feature1 in range(featureNumber):
        for feature2 in range(featureNumber):
            if MI[feature2][feature1] != None:
                MI[feature1][feature2]= MI[feature2][feature1]
            else:
                MI[feature1][feature2] = mutual_info_score(discretizedData[:, feature1], discretizedData[:, feature2])
    return np.array(MI)


def learningMatrix(X,y,knn):

    '''
    Calculate learning matrix using formula (12) from saute article.
    '''

    kNN = NearestNeighbors(n_neighbors=knn)
    kNN.fit(X)
    _, indices = kNN.kneighbors(X)
    weights = np.arange(knn, 0, -1).reshape(knn, 1)

    L = np.sum(y[indices]*weights, axis=1)

    return L

def calculateEntropy(data,D,meanMatrix, stdMatrix, labelsProbs):

    observationNo = data.shape[0]
        
    expandedData = data[None, :, :]       # (1, n, m)

    meanMatrixC = meanMatrix.T[:,None,:]
    stdMatrixC = stdMatrix.T[:,None,:]

    stdSafe = np.where(stdMatrixC > 0, stdMatrixC, 1) 
    xStandardized = np.where(stdMatrixC > 0, (expandedData - meanMatrixC) / stdSafe, 0)
    probs = np.where(stdMatrixC > 0 , (1 / (np.sqrt(2 * np.pi) * stdSafe)) * np.exp(-0.5 * xStandardized**2), 0)

    probsLabels = labelsProbs[:, None, None]

    denominator = np.sum(probsLabels*probs, axis=0)
    denominator = np.where(denominator != 0, denominator, 1)
    newProbs = probsLabels*probs/denominator

    ProbsSafe = np.where(newProbs > 0, newProbs, 1)
    newProbsSafe = np.where(newProbs > 0, ProbsSafe*np.log2(ProbsSafe), 0) 
    entropy = - (np.sum(np.sum(newProbsSafe, axis=0), axis=0)) /observationNo

    return entropy, probs

def calculateCovariance(X, D):

    _,labelNo = D.shape
    N, _ = X.shape

    covMatricies = [None for _ in range(labelNo)]
    for l in range(labelNo):

        mask = D[:, l].astype(bool)
        newX = X[mask, :]

        if newX.shape[0] > 1:
            mean = newX.mean(axis=0)
            newX_centered = newX - mean
            covMat = (newX_centered.T @ newX_centered) / (newX.shape[0] - 1)
        elif newX.shape[0] <= 1:
            covMat = np.eye(X.shape[1]) * 1e-6


        covMatricies[l] = covMat

    return np.array(covMatricies)
device
def calculateCMI2(X, meanArray, covArray, probs, labelsProb):

    obsNo,featuresNo = X.shape
    labelNo = covArray.shape[0]

    CMI = torch.zeros((featuresNo, featuresNo), device = device)

    for i in range(featuresNo):

        xi = X[:, i].unsqueeze(1).unsqueeze(0) # shape [1,N, 1]
        meani = meanArray[i,:].unsqueeze(-1).unsqueeze(-1) # shape [L, 1, 1]
        covii = covArray[:,i, i].unsqueeze(-1).unsqueeze(-1) # shape [L, 1,1]

        pi = probs[:, :, i].unsqueeze(-1)  # shape [L,N, 1]

        for j in range(featuresNo):

            if i >= j:
                continue

            xj = X[:, j].unsqueeze(0).unsqueeze(0)  # shape [1,1, N]
            meanj = meanArray[j,:].unsqueeze(-1).unsqueeze(-1) # shape [L, 1, 1]
            covjj = covArray[:,j, j].unsqueeze(-1).unsqueeze(-1) # shape [L, 1,1]
            covij = covArray[:,i, j].unsqueeze(-1).unsqueeze(-1) # shape [L, 1,1]

            pj = probs[:, :, j].unsqueeze(1)  # shape [L, 1,N]


            mnorm = calculateMutltivariateNormal(xi, xj,meani, meanj, covii, covjj, covij) # shape [F,F,chunk1,chunk2]
                   
            temp = mnorm * (torch.log2(mnorm) - torch.log2(pi) - torch.log2(pj))

            temp = torch.nan_to_num(temp, nan=0.0, posinf=0.0, neginf=0.0)

            cmi = (((temp.sum(-1).sum(-1)) * labelsProb).sum(-1)) / (obsNo**2)

            CMI[i,j] = cmi
            CMI[j,i] = cmi  

    return CMI


def calculateMutltivariateNormal(X0, X1, mean0, mean1, cov00,cov11,cov01):

    det = cov00*cov11- cov01*cov01

    inv00 = 1/det* cov11
    inv11 = 1/det* cov00
    inv01 = -1/det* cov01

    X0Hat = X0- mean0
    X1Hat = X1- mean1

    arg = -0.5*((X0Hat**2)*inv00 + 2*inv01*X0Hat*X1Hat + (X1Hat**2)*inv11)

    prob = 1/(2*torch.pi*torch.sqrt(det)) * torch.exp(arg)

    return prob

def conditionalMutInfo(data, meanArray, covArray, labels, probs, labelsProb):

    N,featuresNo = data.shape

    CMI = [[0 for _ in range(featuresNo)] for __ in range(featuresNo)]
    

    for feature1 in range(featuresNo):
            
        for feature2 in range(featuresNo):

            if feature1 >= feature2:
                continue
            

            X1 = data[:,feature1]


            X2 = data[:,feature2]


            X1mesh, X2mesh = np.meshgrid(X1, X2, indexing='ij') 
            X1X2 = np.column_stack([X1mesh.ravel(), X2mesh.ravel()])

            cmi=0
            for label in labels:

                cov = covArray[label]

                probX1 = probs[label,:, feature1]    
                probX2 = probs[label,:, feature2]
                
                probX1mesh, probX2mesh = np.meshgrid(probX1, probX2, indexing='ij') 
                probX1X2 = np.column_stack([probX1mesh.ravel(), probX2mesh.ravel()])

                meanX1X2 = meanArray[[feature1, feature2], label]

                covX1X2 = cov[np.ix_([feature1, feature2],[feature1, feature2])]

                probsCond = calculateMutltivariateNormal(X1X2, mean=meanX1X2, cov=covX1X2)
                    
                with np.errstate(divide='ignore', invalid='ignore'):
                    vals = probsCond*np.log2(probsCond/np.prod(probX1X2,axis=1))
                    safeVals = np.where(np.isfinite(vals), vals, np.nan)
                    result = np.nansum(safeVals) * labelsProb[label]
                cmi += result
                    
            CMI[feature1][feature2] = cmi
        
    for i in range(featuresNo):
        for j in range(i+1, featuresNo):
            CMI[j][i] = CMI[i][j]
    
    return np.array(CMI)