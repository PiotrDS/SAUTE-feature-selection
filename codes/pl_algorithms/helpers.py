import numpy
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif

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
            Y = ipalUpdate(X=newTrainingSet, y=Y, knn=kNNIpal, alpha=alphaIpal, iterNo=20)

        elif learningType == 'IPALfrac':
            L = ipalUpdate(X=newTrainingSet, y=Y, knn=kNNIpal, alpha=alphaIpal, iterNo=20)

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




def learningMatrix(newX,y,knn):

    kNN = NearestNeighbors(n_neighbors=knn)
    kNN.fit(newX)
    _, indices = kNN.kneighbors(newX)
    weights = np.arange(knn, 0, -1).reshape(knn, 1)

    L = np.sum(y[indices]*weights, axis=1)

    return L




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


def calculateEntropy(data,D,meanMatrix, stdMatrix, labelsProbs):

    observationNo = data.shape[0]
        
    expandedData = data[None, :, :]       # (1, n, m)
    expandedD = D.T[:, :, None]     # (q, n, 1)

    # dataNew = expandedData * expandedD  # (q, n, m) niepotrzebne
    meanMatrixC = meanMatrix.T[:,None,:]
    stdMatrixC = stdMatrix.T[:,None,:]


    stdSafe = np.where(stdMatrixC > 0, stdMatrixC, 1) 
    xStandardized = np.where(stdMatrixC > 0, (expandedData - meanMatrixC) / stdSafe, 0)
    probs = np.where(stdMatrixC > 0 , (1 / (np.sqrt(2 * np.pi) * stdSafe)) * np.exp(-0.5 * xStandardized**2), 0)

    #probs = probs*expandedD #ważne

    probsLabels = labelsProbs[:, None, None]


    denominator = np.sum(probsLabels*probs, axis=0)
    denominator = np.where(denominator != 0, denominator, 1)
    newProbs = probsLabels*probs/denominator


    ProbsSafe = np.where(newProbs > 0, newProbs, 1)
    newProbsSafe = np.where(newProbs > 0, ProbsSafe*np.log2(ProbsSafe), 0) 
    entropy = - (np.sum(np.sum(newProbsSafe, axis=0), axis=0)) /observationNo

    #return probs, newProbs, entropy
    return entropy, probs

def condMutInfo(data, meanArray, covArray, labels, probs, labelsProb):
    N, featuresNo= data.shape

    itemsize    = data.dtype.itemsize
    L2          = 256 * 1024
    usable_cache = int(L2 * 0.5)
    b_max       = int((usable_cache / (4 * itemsize))**0.5)
    block       = max(4, (b_max // 4) * 4)

    print('itemsize: ', itemsize)
    print('usable_cache: ', usable_cache)
    print('b_max: ', b_max)
    print('block: ', block  )

    print('rozmiar: ', block)
    print('petle: ', N// block)

    CMI = [[0 for _ in range(featuresNo)] for __ in range(featuresNo)]

    for feature1 in range(featuresNo):
        X1 = data[:, feature1]
        for feature2 in range(feature1+1, featuresNo):
            X2 = data[:, feature2]
            cmi = 0.0

            # 1) przygotuj marginały raz dla każdej etykiety
            margX1 = {label: probs[label, :, feature1] for label in labels}
            margX2 = {label: probs[label, :, feature2] for label in labels}

            # 2) blokowe obliczenia
            for i in range(0, N, block):
                X1Block = X1[i : i+block]
                for j in range(0, N, block):
                    X2Block = X2[j : j+block]

                    # wspólny mesh dla wartości
                    X1mesh, X2mesh = np.meshgrid(X1Block, X2Block, indexing='ij')
                    X1X2 = np.column_stack([X1mesh.ravel(), X2mesh.ravel()])

                    for label in labels:
                        # ścieżki marginałów z poprawnym slicingiem
                        p1b = margX1[label][i : i+block]
                        p2b = margX2[label][j : j+block]
                        p1m, p2m = np.meshgrid(p1b, p2b, indexing='ij')
                        p1p2 = np.column_stack([p1m.ravel(), p2m.ravel()])

                        meanXY = meanArray[[feature1, feature2], label]
                        covXY  = covArray[label][np.ix_((feature1, feature2),(feature1, feature2))]

                        pCond = calculateMutltivariateNormal(X1X2, mean=meanXY, cov=covXY)

                        with np.errstate(divide='ignore', invalid='ignore'):
                            vals = pCond * np.log2(pCond / np.prod(p1p2, axis=1))
                            safe = np.where(np.isfinite(vals), vals, 0.0)
                            cmi += np.nansum(safe) * labelsProb[label]

            CMI[feature1][feature2] = cmi
            CMI[feature2][feature1] = cmi

    return np.array(CMI)


def conditionalMutInfo(data, meanArray, covArray, labels, probs, labelsProb):

    N,featuresNo = data.shape

    CMI = [[0 for _ in range(featuresNo)] for __ in range(featuresNo)]
    

    for feature1 in range(featuresNo):
        #X1masked = X1[D[:,label]]
            
        #probX1 = prob[:, feature1]
        #probX1masked = probX1[D[:,label]]
            
        for feature2 in range(featuresNo):

            if feature1 >= feature2:
                continue
            

            X1 = data[:,feature1]


            X2 = data[:,feature2]
            #X2masked = X2[D[:,label]]
            #probX2masked = probX2[D[:,label]]


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
                    # niepotrzebne szczegolne przypadki juz ogarniete
                    vals = probsCond*np.log2(probsCond/np.prod(probX1X2,axis=1))
                    safeVals = np.where(np.isfinite(vals), vals, np.nan)
                    result = np.nansum(safeVals) * labelsProb[label]
                cmi += result
                    
            CMI[feature1][feature2] = cmi
        
    for i in range(featuresNo):
        for j in range(i+1, featuresNo):
            CMI[j][i] = CMI[i][j]
    
    return np.array(CMI)

def alternativeMutInfo(cov, std):

    corrScore = np.empty(shape=cov.shape)

    for label in range(cov.shape[0]):
        s = std[:,label]
        sXY = np.outer(s, s)
        score = cov[label, :, :] / sXY
        np.fill_diagonal(score,0)
        corrScore[label,:,:] = score
        
    corrScore = -0.5 * np.log2(1-(corrScore**2))
    #corrScore[np.isposinf(corrScore)] = -np.inf
    
    return np.sum(corrScore, axis=0)



def conditionalMutInfo_gpu(data, meanArray, covArray, probs, labelsProb):
    N, F = data.shape
    L = labelsProb.shape[0]

    labels = range(L)

    platforms = cl.get_platforms()
    gpu_devs  = platforms[0].get_devices(device_type=cl.device_type.GPU)
    ctx       = cl.Context(devices=gpu_devs)
    queue     = cl.CommandQueue(ctx)
    mf        = cl.mem_flags

    # 2) Kompilacja kernela tylko raz
    kernel_src = """
    __kernel void cmi_kernel(
        int           N,
        __global const float *X1,
        __global const float *X2,
        __global const float *p1,
        __global const float *p2,
        float         labelProb,
        float         mean1,
        float         mean2,
        float         inv00,
        float         inv01,
        float         inv11,
        float         det_sqrt,
        __global float       *out)
    {
        int i   = get_global_id(0);
        int j   = get_global_id(1);
        int gid = j + i * N;

        float x1 = X1[i] - mean1;
        float x2 = X2[j] - mean2;
        float q  = inv00*x1*x1 + 2.0f*inv01*x1*x2 + inv11*x2*x2;
        float expo = exp(-0.5f * q);
        float norm = 2.0f * 3.141592653589793f * det_sqrt;
        float p12  = expo / norm;

        float v1 = p1[i], v2 = p2[j];
        float val = 0.0f;
        if (p12 > 0.0f && v1 > 0.0f && v2 > 0.0f)
            val = p12 * log2(p12 / (v1 * v2));

        out[gid] = val * labelProb;
    }
    """
    prg = cl.Program(ctx, kernel_src).build()

    X_bufs = [
        cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                  hostbuf=data[:, f].astype(np.float32))
        for f in range(F)
    ]

    p1_buf  = cl.Buffer(ctx, mf.READ_ONLY,  size=N * 4)
    p2_buf  = cl.Buffer(ctx, mf.READ_ONLY,  size=N * 4)
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=N * N * 4)

    tmp = np.empty(N * N, dtype=np.float32)
    CMI = np.zeros((F, F), dtype=np.float32)

    global_size = (N, N)
    local_size  = None  

    for f1 in range(F):
        for f2 in range(f1 + 1, F):
            total = 0.0
            X1_buf = X_bufs[f1]
            X2_buf = X_bufs[f2]

            for lab in labels:

                p1 = probs[lab, :, f1].astype(np.float32)
                p2 = probs[lab, :, f2].astype(np.float32)
                cl.enqueue_copy(queue, p1_buf, p1)
                cl.enqueue_copy(queue, p2_buf, p2)

                cov = covArray[lab][np.ix_([f1, f2], [f1, f2])].copy()
                eps = 1e-6
                cov[0,0] += eps; cov[1,1] += eps

                det = cov[0,0]*cov[1,1] - cov[0,1]*cov[1,0]
                if det <= 0:
                    continue
                det_sqrt = np.float32(np.sqrt(det))
                inv00    = np.float32( cov[1,1] / det )
                inv01    = np.float32(-cov[0,1] / det )
                inv11    = np.float32( cov[0,0] / det )

                evt_ker = prg.cmi_kernel(
                    queue, global_size, local_size,
                    np.int32(N),
                    X1_buf, X2_buf,
                    p1_buf, p2_buf,
                    np.float32(labelsProb[lab]),
                    np.float32(meanArray[f1, lab]),
                    np.float32(meanArray[f2, lab]),
                    inv00, inv01, inv11, det_sqrt,
                    out_buf
                )
                evt_copy = cl.enqueue_copy(queue, tmp, out_buf, wait_for=[evt_ker])
                evt_copy.wait()

                tmp = np.nan_to_num(tmp, nan=0.0, posinf=0.0, neginf=0.0)
                total += tmp.sum()

            CMI[f1, f2] = CMI[f2, f1] = total

    return CMI





def cmi(data, D,meanArray, covArray, labels, probs, labelsProb):


    featuresNo = data.shape[1]
    labelsNo = len(labels)


    CMI = [[0 for _ in range(featuresNo)] for __ in range(featuresNo)]
    

    for label in labels:

        cov = covArray[label]
        prob = probs[label]      

        for feature1 in range(featuresNo):
            X1 = data[:,feature1]
            X1masked = X1[D[:,label]]
            
            probX1 = prob[:, feature1]
            probX1masked = probX1[D[:,label]]
            
            for feature2 in range(featuresNo):

                if feature1 >= feature2:
                    continue

                X2 = data[:,feature2]
                X2masked = X2[D[:,label]]
                
                probX2 = prob[:, feature2]
                probX2masked = probX2[D[:,label]]

                X1mesh, X2mesh = np.meshgrid(X1masked, X2masked, indexing='ij') 
                X1X2 = np.column_stack([X1mesh.ravel(), X2mesh.ravel()])  

                probX1mesh, probX2mesh = np.meshgrid(probX1masked, probX2masked, indexing='ij') 
                probX1X2 = np.column_stack([probX1mesh.ravel(), probX2mesh.ravel()])

                meanX1X2 = meanArray[[feature1, feature2], label]

                covX1X2 = cov[np.ix_([feature1, feature2],[feature1, feature2])]

                probsCond = calculateMutltivariateNormal(X1X2, mean=meanX1X2, cov=covX1X2)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    # niepotrzebne szczegolne przypadki juz ogarniete
                    vals = probsCond*np.log2(probsCond/np.prod(probX1X2,axis=1))
                    safeVals = np.where(np.isfinite(vals), vals, np.nan)
                    result = np.nansum(safeVals) * labelsProb[label]
                
                CMI[feature1][feature2] += result
    
    for i in range(featuresNo):
        for j in range(i+1, featuresNo):
            CMI[j][i] = CMI[i][j]
    
    return np.array(CMI)

def discretizeData(data):

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


def convertSparseMatrixIntoVector(sparseMat):   

    return np.argmax(sparseMat, axis=1)


def calculateCovariance(X, D):

    _,labelNo = D.shape
    N, featureNo = X.shape

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

def calculateMutltivariateNormal(X, mean, cov):

    eps = 10e-7
    cov +=np.eye(cov.shape[0]) * eps

    det = cov[0,0]*cov[1,1] - cov[0,1]*cov[1,0]

    invCov = 1/det* np.array([[cov[1,1], -cov[0,1]],[-cov[1,0], cov[0,0]]])

    xmean = X- mean

    arg = -0.5*(
        xmean[:, 0] * (invCov[0, 0] * xmean[:, 0] + invCov[0, 1] * xmean[:, 1]) +
        xmean[:, 1] * (invCov[1, 0] * xmean[:, 0] + invCov[1, 1] * xmean[:, 1])
    )

    prob = 1/(2*np.pi*np.sqrt(det)) * np.exp(arg)

    return prob


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

def convertSparseMatrixIntoVector(sparseMat):   

    return np.argmax(sparseMat, axis=1)


def ipalUpdate(X,y, knn=10, alpha=0.95, iterNo=20):

    observationNo = X.shape[0]

    # Identify nearest neighbours for each observation
    kNN = NearestNeighbors(n_neighbors=knn)
    kNN.fit(X)
    distances, indices = kNN.kneighbors()


    # initialize weight matrix
    W = [[0 for _ in range(observationNo)] for __ in range(observationNo)]

    for observationIdx, idx in enumerate(indices):

        weightsNew = solveQuadr(observation=X[observationIdx, :], neighborsMatrix=X[idx, :])
        # do sprawdzenia X[idx, :] 


        for j, i in enumerate(idx):
            W[i][observationIdx] = weightsNew[j,0]
    W = np.array(W)
    
    # update W matrix

    invD = np.diag(1/np.sum(W, axis=0))

    H = W @ invD

    # create labeling confidence matrix


    yCount = np.sum(y, axis=1, keepdims=True)
    P = np.divide(y,yCount) 

    y01 = y > 0 

    F = P

    for i in range(iterNo):

        F = alpha * H.T @ F + (1-alpha)*P

        #rescale F matrix
        F =F * y01 
        F = F / np.sum(F, axis=1, keepdims=True)

    sum_F = np.sum(F, axis=0, keepdims=True)
    sum_P = np.sum(P, axis=0, keepdims=True)

    ratio = np.divide(sum_P, sum_F, out=np.zeros_like(sum_P), where=sum_F != 0)

    finalF = ratio * F


    return finalF

def solveQuadr(observation, neighborsMatrix):
    N, d = neighborsMatrix.shape 

    P = neighborsMatrix @ neighborsMatrix.T  
    q = (-neighborsMatrix @ observation).reshape(-1, 1)
    G = -np.identity(N)
    h = np.zeros((N, 1))
    
    weights = cp.Variable((N, 1)) 

    objective = cp.Minimize(cp.quad_form(weights, P) + 2 * q.T @ weights)

    constraints = [G @ weights <= h]

    problem = cp.Problem(objective, constraints)


    try:
        result = problem.solve(solver=cp.OSQP, verbose=False)
    
    except Exception as e:
        weights = cp.Variable((N, 1), pos=True) 

        objective = cp.Minimize(cp.quad_form(weights, cp.psd_wrap(P)) + 2 * q.T @ weights)

        constraints = [G @ weights <= h]

        problem = cp.Problem(objective, constraints)
        result = problem.solve(cp.ECOS, verbose=False)  

    return weights.value  


def predictIpalW(XTrain,XTest, y, nn):

    Y =  np.argmax(y, axis=1)
    observationNo = XTest.shape[0]

    # Identify nearest neighbours for each observation
    kNN = NearestNeighbors(n_neighbors=nn)
    kNN.fit(XTrain)

    distances, indices = kNN.kneighbors(XTest)

    yPred = [None for _ in range(observationNo)]

    for observationIdx, idx in enumerate(indices):
        weights = solveQuadr(observation=XTest[observationIdx, :], neighborsMatrix=XTrain[idx, :])

        labelsScores = {label: XTest[observationIdx].copy() for label in np.unique(Y[idx])}
        for (i,id) in enumerate(idx):
            sumLabels = weights[i,:] * XTrain[id, :]
            labelsScores[Y[id]] -= sumLabels
        for label,score in labelsScores.items():
            labelsScores[label] = np.sum(score*score)
        predLabel = min(labelsScores, key=labelsScores.get) 

        yPred[observationIdx] = predLabel

    return np.array(yPred)


def predictIpal(XTrain,XTest, yTrain, nn,alpha, iterNo=15):

    YIpal, kNN = ipal(XTrain, yTrain, nn, alpha, iterNo)


    distances, indices = kNN.kneighbors(XTest)

    observationNo = XTest.shape[0]    
    labelsNo = yTrain.shape[1]

    yPred = [None for _ in range(observationNo)]

    for observationIdx, idx in enumerate(indices):
        weights = solveQuadr(observation=XTest[observationIdx, :], neighborsMatrix=XTrain[idx, :])

        labelsScores = {label: XTest[observationIdx].copy() for label in np.unique(YIpal[idx])}
        for (i,id) in enumerate(idx):
            sumLabels = weights[i,:] * XTrain[id, :]
            labelsScores[YIpal[id]] -= sumLabels
        for label,score in labelsScores.items():
            labelsScores[label] = np.sum(score*score)
        predLabel = min(labelsScores, key=labelsScores.get) 

        yPred[observationIdx] = predLabel

    return np.array(yPred), YIpal
    

def ipal(X,y, knn=10, alpha=0.95, iterNo=20):

    observationNo = X.shape[0]

    # Identify nearest neighbours for each observation
    kNN = NearestNeighbors(n_neighbors=knn)
    kNN.fit(X)
    distances, indices = kNN.kneighbors()


    # initialize weight matrix
    W = [[0 for _ in range(observationNo)] for __ in range(observationNo)]

    for observationIdx, idx in enumerate(indices):

        weightsNew = solveQuadr(observation=X[observationIdx, :], neighborsMatrix=X[idx, :])
        # do sprawdzenia X[idx, :] 


        for j, i in enumerate(idx):
            W[i][observationIdx] = weightsNew[j,0]
    W = np.array(W)
    
    # update W matrix

    invD = np.diag(1/np.sum(W, axis=0))

    H = W @ invD

    # create labeling confidence matrix
    yCount = np.sum(y, axis=1, keepdims=True)
    P = np.divide(y,yCount) 

    F = P

    for i in range(iterNo):

        F = alpha * H.T @ F + (1-alpha)*P

        #rescale F matrix
        F =F * y
        F = F / np.sum(F, axis=1, keepdims=True)

    Yprobs = np.sum(P, axis=0, keepdims=True) / np.sum(F, axis=0, keepdims=True) * F

    Y = np.argmax(Yprobs, axis=1)

    return Y, kNN

def calcMI(X,yPl, yTrue):
    '''
    zwraca infromacje wzajemną między dla każdej l - klasy:
     1. X ~ (Y_pl = l) (all)
     2. (X | Y = l) ~ (Y_pl = l | Y = l) (subs)
    '''
    labelNo = yTrue.shape[1]
    all = [None for i in range(labelNo)]
    subs = [None for i in range(labelNo)]
    for label in range(labelNo):
        Y = np.logical_not(yTrue)[:,label]
        all[label]=mutual_info_classif(X, yPl[:,label])
        subs[label] = mutual_info_classif(X[Y,:], yPl[Y,label])
    return np.array(all), np.array(subs)


def RS(X, numberOfFeaturesToChoose):
    selected_columns = np.random.choice(X.columns, size=numberOfFeaturesToChoose, replace=False)
    X_selected = X[selected_columns]
    return X_selected