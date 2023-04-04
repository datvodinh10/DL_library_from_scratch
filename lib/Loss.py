import numpy as np

def MSELoss(y_true,y_pred):
    return np.mean(np.square(y_true-y_pred))

def CrossEntropyLoss(y_true,y_pred):
    pass
# def 