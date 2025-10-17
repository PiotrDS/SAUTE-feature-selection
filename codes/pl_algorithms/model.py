import numpy as np
import torch
from .saute import saute, sauteGPU
from .knn import knnPartialPredict
from .ipal import predictIpal

class Model:

    def __init__(self, X, y_pl) -> None:
        self.X = X
        self.y_pl = y_pl
        self.y_confidence = y_pl /  np.sum(y_pl, axis=1)[:, np.newaxis]
        self.selected_vars = list(range(X.shape[1])) # initialize with full set of variables

    def select_saute(self, vars_no, knn = 8, knn_Ipal=8,alpha=0.6,
          alpha_Ipal=0.9,criterium='original',learning_type='original', 
          max_iter=20, use_cuda = False) -> np.array:
        
        if use_cuda:
            
            if torch.cuda.is_available():
                A, y_updated = sauteGPU(self.X, self.y_pl, vars_no,vars_no, kNN=knn, kNNIpal=knn_Ipal, alpha=alpha,
                                        kNNIpal=alpha_Ipal,criterium=criterium,learningType=learning_type, 
                                        max_iter=max_iter)
            else:
                raise RuntimeError('CUDA-compatible device not detected.')
        
        else:
            
            A, y_updated = saute(self.X, self.y_pl, vars_no, kNN=knn, kNNIpal=knn_Ipal, alpha=alpha,
                                        kNNIpal=alpha_Ipal,criterium=criterium,learningType=learning_type, 
                                        max_iter=max_iter)
        
        self.y_confidence = y_updated
        self.selected_vars = A 
        return self.X[:, self.selected_vars]
        
    def knn_predict(self, X_new, knn=8, weight_type="order", use_selected_vars=False) -> np.array:
        if use_selected_vars:
            predictions = knnPartialPredict(self.X[:, self.selected_vars], X_new[:, self.selected_vars], self.y_pl, knn, weight_type)
        else:
            predictions = knnPartialPredict(self.X, X_new, self.y_pl, knn, weight_type)

        return predictions
        

    def ipal_predict(self, X_new, knn=8, alpha=0.9, iter_no=20, use_selected_vars=False) -> np.array:
        if use_selected_vars:
            predictions, _ = predictIpal(self.X[:, self.selected_vars], X_new[:, self.selected_vars], self.y_pl, knn, alpha, iter_no)
        else:
            predictions, _ = predictIpal(self.X, X_new, self.y_pl, knn, alpha, iter_no)
        
        return predictions