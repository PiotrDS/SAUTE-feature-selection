import numpy as np
import torch
from .helpers import calculateMutualInformation, calculateCovariance, calculateEntropy, alternativeMutInfo, learningMatrix, calculateCMI2
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

    sauteScores = {i:{'selectedFeatures':None,
               'yUpdated': None} for i in range(maxIter)}
    
    # Extract features names
    features = range(0,X.shape[1])

    # Number of labels
    numberOfLables = y.shape[1]
    targetLabels = range(numberOfLables)


    # Number of features and observations
    numberOfObservations = X.shape[0]
    numberOfFeatures = X.shape[1]


    # calculate Mutual Information between variables
    MI = calculateMutualInformation(X)


    # create labeling confidence matrix
    yCount = np.sum(y, axis=1)[:, np.newaxis]
    Y = np.divide(y,yCount, dtype='float32')


    for iteration in range(maxIter):

        print(f'iteration number: {iteration}',flush=True ,end=' ')


        A = set()
        F_A = set(features) # set difference of F and A =: F \ A

        previousA = set()

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
            #CMI = cmi(X,D, mean, cov, np.arange(len(targetLabels),dtype='int64'), probs, labelsProbability)
            #CMI = condMutInfo(X, mean, cov, np.arange(len(targetLabels),dtype='int64'), probs, labelsProbability)
            #conditionalMutInfo_numba

            # idx = np.random.choice(numberOfObservations, size=saplesNo, replace=False)
            
            # newX = X[idx]
            # newProbs = probs[:, idx,:]

            CMI = alternativeMutInfo(cov, std)

        
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
                        
                    #featuresEntropy[chosenFeature] = np.inf

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
                        
                    #featuresEntropy[chosenFeature] = np.inf

                    scores = -featuresEntropy - (1/len(A))*sumMI + (1/len(A))*sumCMI

                    scores[list(A)] = -np.inf

                    chosenFeature = np.argmax(scores)
            
            # add chosen feature
            A.add(int(chosenFeature))


        # after the above iteration A contains numberOfFeatures features

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

        # update history
        sauteScores[iteration]['selectedFeatures'] = list(A)
        sauteScores[iteration]['yUpdated'] = np.copy(Y)

        if previousA == A:
            break

        previousA = A

    print('')
    
    return sauteScores


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

    sauteScores = {i:{'selectedFeatures':None,
               'yUpdated': None} for i in range(maxIter)}

    # Extract features names
    features = range(0,X.shape[1])

    # Number of labels
    numberOfLables = y.shape[1]
    targetLabels = range(numberOfLables)


    # Number of features and observations
    numberOfObservations = X.shape[0]
    numberOfFeatures = X.shape[1]


    # calculate Mutual Information between variables
    MI = calculateMutualInformation(X)


    # create labeling confidence matrix
    yCount = np.sum(y, axis=1)[:, np.newaxis]
    Y = np.divide(y,yCount, dtype='float32')

    Xtens = torch.from_numpy(X).to(dtype=torch.float32, device=device)


    for iteration in range(maxIter):

        print(f'iteration number: {iteration}',flush=True ,end=' ')


        A = set()
        F_A = set(features) # set difference of F and A =: F \ A

        previousA = set()

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

            meanTens = torch.from_numpy(mean).to(dtype=torch.float32, device=device)
            covTens= torch.from_numpy(cov).to(dtype=torch.float32, device=device)
            probsTens = torch.from_numpy(probs).to(dtype=torch.float32, device=device)
            labelsProbsTens =  torch.from_numpy(labelsProbability).to(dtype=torch.float32, device=device)

            CMItens = calculateCMI2(Xtens, meanTens, covTens, probsTens, labelsProbsTens)


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


        # after the above iteration A contains numberOfFeatures features

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


        # update history
        sauteScores[iteration]['selectedFeatures'] = list(A)
        sauteScores[iteration]['yUpdated'] = np.copy(Y)

        if previousA == A:
            break

        previousA = A

    print('')
   
    return sauteScores