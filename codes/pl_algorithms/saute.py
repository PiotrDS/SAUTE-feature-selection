import numpy as np
import torch
from .helpers import calculateMutualInformation, calculateCovariance, calculateEntropy, conditionalMutInfo, learningMatrix, calculateCMI2
from .ipal import ipal

def saute(X,y, numberOfChosenFeatures = 3, kNN = 4, kNNIpal=8,alpha=0.6,
          alphaIpal=0.9,criterium='original',learningType='original', 
          maxIter=20):

    '''
    Function saute is implementing feature selection 
    algorithm - SAUTE

    input data: 
     X - dataset
     y - target label
     numberOfChosenFeatures - number of features to choose for algorithm
     kNN - number of nearest neighbours for kNN algorithm
     alpha - learning rate
    '''

    assert criterium in ('JMI','CIFE', 'original'), f'wrong criterium chosen'
    assert learningType in ('IPALall','IPALfrac' , 'original'), f'wrong learningType chosen'

    # Step I: Initialize labeling confidence matrix and calculate mutual information
    # between features

    # Extract features names
    features = range(0,X.shape[1])

    # Number of features and observations
    numberOfObservations = X.shape[0]

    # calculate Mutual Information between variables
    MI = calculateMutualInformation(X)

    # create labeling confidence matrix
    yCount = np.sum(y, axis=1)[:, np.newaxis]
    Y = np.divide(y,yCount, dtype='float32')

    # Step II: Iterative selection of a subset of variables and updating of the labeling confidence matrix

    previousA = set()

    for iteration in range(maxIter):
        
        A = set() # start with empty set
        F_A = set(features) # set difference of F and A =: F \ A

        # create affiliation matrix D:
        # D[i,j] = 1 if Y[i,j] >= 1/n_i, where n_i is a number of labels in his candidate label set  
        # D[i,j] = 0, otherwise
        # in other words matrix D in ith row contains information
        # which label might be its real one
        D = (Y >= 1/yCount) 

        # calculate mean and standard devation 
        sumD = np.sum(D, axis=0)
        mean = np.zeros((X.shape[1], D.shape[1]))
        mask = sumD > 0
        mean[:, mask] = (X.T @ D[:, mask]) / sumD[mask]

        

        if criterium == 'original':
            # calculate standard devation 
            sumD = np.sum(D, axis=0) 
            std = np.zeros((X.shape[1], D.shape[1]))

            mask = sumD > 1  

            numerator = ((X**2).T @ D[:, mask]) - ((X.T @ D[:, mask])**2 / sumD[mask])
            denominator = sumD[mask] - 1
            std[:, mask] = np.sqrt(numerator / denominator)


        elif criterium == 'CIFE' or criterium == 'JMI':
            # calculate covariance
            cov = calculateCovariance(X, D)
            # extract standard devation from covariance matrix
            std = np.sqrt(np.einsum('...ii->...i', cov)).T
            # calculate conditional MI
            CMI = conditionalMutInfo(cov, std)

        # calculate p(u) = 1/m * sum(i=1,..,m) of Y(i,u), where m = numberOfObservations
        labelsProbability = (np.sum(Y,axis=0))/numberOfObservations
  
        # calculate entropy
        featuresEntropy, _ = calculateEntropy(X, D, mean, std, labelsProbability)


        
        for _ in range(numberOfChosenFeatures):
            
            # update F_A
            F_A = F_A.difference(A)

            # find feature maximizing given criterium
            
            if criterium == 'original':
                if len(A) == 0:
                    chosenFeature = np.argmax(-featuresEntropy)

                else:
                    updatedMI = MI[list(A), :]
                    sumMI = np.sum(updatedMI, axis=0)
                        
                    featuresEntropy[chosenFeature] = np.inf

                    scores = -featuresEntropy - (1/len(A))*sumMI

                    chosenFeature = np.argmax(scores)
            
            elif criterium == 'CIFE':
                if len(A) == 0:
                    chosenFeature = np.argmax(-featuresEntropy)

                else:
                    updatedMI = MI[list(A), :]
                    sumMI = np.sum(updatedMI, axis=0)

                    updatedCMI = CMI[list(A), :]
                    sumCMI = np.sum(updatedCMI, axis=0)
                        
                    scores = -featuresEntropy - sumMI + sumCMI

                    scores[list(A)] = -np.inf

                    chosenFeature = np.argmax(scores)
            
            elif criterium == 'JMI':
                if len(A) == 0:
                    chosenFeature = np.argmax(-featuresEntropy)

                else:
                    updatedMI = MI[list(A), :]
                    sumMI = np.sum(updatedMI, axis=0)

                    updatedCMI = CMI[list(A), :]
                    sumCMI = np.sum(updatedCMI, axis=0)
                        
                    scores = -featuresEntropy - (1/len(A))*sumMI + (1/len(A))*sumCMI

                    scores[list(A)] = -np.inf

                    chosenFeature = np.argmax(scores)
            
            # add chosen feature
            A.add(int(chosenFeature))


        # Step 2a: Update labeling confidence matrix using learning matrix and selected subset of variables

        # Identify the 𝑘 nearest neighbors N(x_i) for all observations in 'smaller' dataset

        newTrainingSet = X[:,list(A)]

        # Calculate the learning matrix L

        if learningType == 'original':
            L = learningMatrix(newTrainingSet, Y, knn=kNN)

            # Update Y

            Y = (1-alpha)*Y + alpha*L

            # scale Y

            Y = Y*y

            Y = Y/np.sum(Y,axis=1)[:, np.newaxis]
        
        elif learningType == 'IPALall':
            _,__, Y = ipal(X=newTrainingSet, y=Y, knn=kNNIpal, alpha=alphaIpal, iterNo=20)

        elif learningType == 'IPALfrac':
            _,__, L = ipal(X=newTrainingSet, y=Y, knn=kNNIpal, alpha=alphaIpal, iterNo=20)

            # Update Y

            Y = (1-alpha)*Y + alpha*L

            # scale Y

            Y = Y*y

            Y = Y/np.sum(Y,axis=1)[:, np.newaxis]


        if previousA == A:
            break #stop if set didn't chang

        previousA = A
    
    return list(A), Y


def sauteGPU(X,y,numberOfChosenFeatures = 3, kNN = 4,
          alpha=0.6,alphaIpal=0.9,criterium='original',
          learningType='original', maxIter=20):
   

    '''
    Function saute is implementing feature selection
    algorithm - SAUTE

    input data:
     X - dataset
     y - target label
     numberOfChosenFeatures - number of features to choose for algorithm
     kNN - number of nearest neighbours for kNN algorithm
     alpha - learning rate
    '''

    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    assert criterium in ('JMI','CIFE', 'original'), f'wrong criterium chosen'
    assert learningType in ('IPALall','IPALfrac' , 'original'), f'wrong learningType chosen'

    # Step I: Initialize labeling confidence matrix and calculate mutual information
    # between features


    # Extract features names
    features = range(0,X.shape[1])

    # Number of features and observations
    numberOfObservations = X.shape[0]

    # calculate Mutual Information between variables
    MI = calculateMutualInformation(X)


    # create labeling confidence matrix
    yCount = np.sum(y, axis=1)[:, np.newaxis]
    Y = np.divide(y,yCount, dtype='float32')

    # transer X to gpu (if available, if don't SAUTE function is recommended)
    Xtens = torch.from_numpy(X).to(dtype=torch.float32, device=device)

    previousA = set()

    # Step II: Iterative selection of a subset of variables and updating of the labeling confidence matrix


    for iteration in range(maxIter):

        A = set()
        F_A = set(features) # set difference of F and A =: F \ A

        # create affiliation matrix D:
        # D[i,j] = 1 if Y[i,j] >= 1/n_i, where n_i is a number of labels in his candidate label set  
        # D[i,j] = 0, otherwise
        # in other words matrix D in ith row contains information
        # which label might be its real one
        D = (Y >= 1/yCount)

        # calculate mean and standard devation
        sumD = np.sum(D, axis=0)
        mean = np.zeros((X.shape[1], D.shape[1]), dtype='float32')
        mask = sumD > 0
        mean[:, mask] = (X.T @ D[:, mask]) / sumD[mask]

           

        if criterium == 'original':
            # calculate standard devation
            sumD = np.sum(D, axis=0)
            std = np.zeros((X.shape[1], D.shape[1]), dtype='float32')

            mask = sumD > 1  

            numerator = ((X**2).T @ D[:, mask]) - ((X.T @ D[:, mask])**2 / sumD[mask])
            denominator = sumD[mask] - 1
            std[:, mask] = np.sqrt(numerator / denominator)

               
            #std = np.sqrt(((X**2).T @ D - ((X.T @ D)**2 / np.sum(D, axis=0)))/(np.sum(D, axis=0)-1))


        elif criterium == 'CIFE' or criterium == 'JMI':
            # calculate covariance
            cov = calculateCovariance(X, D)
            # extract standard devation from covariance matrix
            std = np.sqrt(np.einsum('...ii->...i', cov)).T

        # calculate p(u) = 1/m * sum(i=1,..,m) of Y(i,u), where m = numberOfObservations
        labelsProbability = (np.sum(Y,axis=0))/numberOfObservations

 
        # calculate entropy
        featuresEntropy, probs = calculateEntropy(X, D, mean, std, labelsProbability)



        if criterium == 'CIFE' or criterium=='JMI':
            # tranfer to gpu
            meanTens = torch.from_numpy(mean).to(dtype=torch.float32, device=device)
            covTens= torch.from_numpy(cov).to(dtype=torch.float32, device=device)
            probsTens = torch.from_numpy(probs).to(dtype=torch.float32, device=device)
            labelsProbsTens =  torch.from_numpy(labelsProbability).to(dtype=torch.float32, device=device)
            # calculate conditional mutual information on gpu
            CMItens = calculateCMI2(Xtens, meanTens, covTens, probsTens, labelsProbsTens)
            # transfer back to cpu
            CMI = CMItens.detach().cpu().numpy()


        for _ in range(numberOfChosenFeatures):

           
            # update F_A
            F_A = F_A.difference(A)

            # find feature maximizing criterium
            if criterium == 'original':
                if len(A) == 0:
                    chosenFeature = np.argmax(-featuresEntropy)

                else:
                    updatedMI = MI[list(A), :]
                    sumMI = np.sum(updatedMI, axis=0)
                       
                    featuresEntropy[chosenFeature] = np.inf

                    scores = -featuresEntropy - (1/len(A))*sumMI

                    chosenFeature = np.argmax(scores)
            elif criterium == 'CIFE':
                if len(A) == 0:
                    chosenFeature = np.argmax(-featuresEntropy)

                else:
                    updatedMI = MI[list(A), :]
                    sumMI = np.sum(updatedMI, axis=0)

                    updatedCMI = CMI[list(A), :]
                    sumCMI = np.sum(updatedCMI, axis=0)
                       
                    featuresEntropy[chosenFeature] = np.inf

                    scores = -featuresEntropy - sumMI + sumCMI

                    chosenFeature = np.nanargmax(scores)
            elif criterium == 'JMI':
                if len(A) == 0:
                    chosenFeature = np.argmax(-featuresEntropy)

                else:
                    updatedMI = MI[list(A), :]
                    sumMI = np.sum(updatedMI, axis=0)

                    updatedCMI = CMI[list(A), :]
                    sumCMI = np.sum(updatedCMI, axis=0)
                       
                    featuresEntropy[chosenFeature] = np.inf

                    scores = -featuresEntropy - (1/len(A))*sumMI + (1/len(A))*sumCMI

                    chosenFeature = np.nanargmax(scores)
           
            # add chosen feature
            A.add(int(chosenFeature))


        # Step 2a: Update labeling confidence matrix using learning matrix and selected subset of variables

        # Identify the 𝑘 nearest neighbors N(x_i) for all observations in 'smaller' dataset

        newTrainingSet = X[:,list(A)]

        # Calculate the learning matrix L

        if learningType == 'original':
            L = learningMatrix(newTrainingSet, Y, knn=kNN)

            # Update Y

            Y = (1-alpha)*Y + alpha*L

            # scale Y

            Y = Y*y

            Y = Y/np.sum(Y,axis=1)[:, np.newaxis]
       
        elif learningType == 'IPALall':
            _,__,Y = ipal(X=newTrainingSet, y=Y, knn=kNN, alpha=alphaIpal, iterNo=20)

        if previousA == A:
            break

        previousA = A
   
    return list(A), Y