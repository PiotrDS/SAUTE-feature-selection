import numpy
from .saute import saute, sauteGPU

class Model:

    def __init__(self, X, y_pl) -> None:
        self.X = X
        self.y_pl = y_pl

    def select_saute(self, vars_no) -> numpy.array:
        new_X = saute(self.X, self.y_pl, vars_no)
        return new_X
        
    def pl_knn_predict(self, X_new) -> numpy.array:
        pass

    def ipal_predict(self, X_new) -> numpy.array:
        pass 