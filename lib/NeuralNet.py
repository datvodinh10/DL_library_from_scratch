import numpy as np

def Forward(W1,b1,W2,b2,W3,b3,X):
    O1 = W1 @ X.T + b1
    G1 = np.maximum(0,O1)
    O2 = W2 @ G1 + b2
    G2 = np.maximum(0,O2)
    O3 = W3 @ G2 + b3
    return O1,O2,O3

def Mean(X):
    R = []
    for i in np.arange(X.shape[0]):
        R.append(X[i,:].mean())
    return np.array(R).reshape(-1,1) * 1.0

def Backward(W1,b1,W2,b2,W3,b3,S,Y,O1,O2,O3,lr):
    # print(O3.shape,Y.shape)
    dO3 = (O3-Y.T)
    dG2 =  W3.T.dot(dO3)
    dG1 = W2.T.dot(dG2 * (O2 >= 0))
    dW1 =  (dG1 * (O1 >= 0)).dot(S)
    dB1 = Mean(dG1 * (O1 >= 0))
    dW2 = (dG2 * (O2 >= 0)).dot(np.maximum(0,O1).T)
    dB2 = Mean(dG2 * (O2 >= 0))
    dW3 = dO3.dot(np.maximum(0,O2).T)
    dB3 = Mean(dO3)
    return dW1,dB1,dW2,dB2,dW3,dB3



def MSELoss(y_true,y_pred):
    return np.mean(np.square(y_true-y_pred))

def InitParams(X,y):
    W1 = np.random.uniform(low=-1/np.sqrt(X.shape[1]),high=1/np.sqrt(X.shape[1]),size=(128 ,X.shape[1]))
    b1 = np.zeros((128,1)) * 1.0
    W2 = np.random.uniform(low=-1/np.sqrt(128),high=1/np.sqrt(128),size=(256,128))
    b2 = np.zeros((256,1)) * 1.0
    W3 = np.random.uniform(low=-1/np.sqrt(256),high=1/np.sqrt(256),size=(y.shape[1],256))
    b3 = np.zeros((1,1)) * 1.0
    return W1,b1,W2,b2,W3,b3

def FitModel(X,y,n_iter=10,batch_size=16,lr=1e-4,UpdateParams=None,print_stat=True):
    W1,b1,W2,b2,W3,b3 = InitParams(X,y)
    for _ in range(n_iter):
        W1,b1,W2,b2,W3,b3 = UpdateParams(W1,b1,W2,b2,W3,b3,X,y,batch_size=batch_size,lr=lr)
        if print_stat:
            if _ % 50 == 0:
                print('Epoch ',_, 'Loss: ',MSELoss(y.reshape(1,-1),Forward(W1,b1,W2,b2,W3,b3,X)[-1]))
    return W1,b1,W2,b2,W3,b3

