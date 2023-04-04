from lib.NeuralNet import *
def OptimizerMomentum(W1,b1,W2,b2,W3,b3,dW1,dB1,dW2,dB2,dW3,dB3,vW1,vb1,vW2,vb2,vW3,vb3,lr,beta):
    vW1,vb1,vW2,vb2,vW3,vb3 = (beta * vW1 + dW1 , beta * vb1 + dB1, beta * vW2 + dW2, beta * vb2 + dB2, beta * vW3 + dW3, beta * vb3 + dB3)
    W1,b1,W2,b2,W3,b3 = W1 - lr * vW1, b1 - lr * vb1, W2 - lr * vW2, b2 - lr * vb2, W3 - lr * vW3, b3 - lr * vb3
    return W1,b1,W2,b2,W3,b3,vW1,vb1,vW2,vb2,vW3,vb3
        
def Momentum(W1,b1,W2,b2,W3,b3,S,Y,batch_size=16,lr=1e-4,beta=0.99,epoch=0):
    n_samples = S.shape[0]
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    S,Y = S[idx],Y[idx]
    vW1,vb1,vW2,vb2,vW3,vb3 = np.zeros_like(W1),np.zeros_like(b1),np.zeros_like(W2),np.zeros_like(b2),np.zeros_like(W3),np.zeros_like(b3)
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i + batch_size, n_samples)
        s,y =  S[begin:end] , Y[begin:end]
        O1,O2,O3 = Forward(W1,b1,W2,b2,W3,b3,s)
        dW1,dB1,dW2,dB2,dW3,dB3 = Backward(W1,b1,W2,b2,W3,b3,s,y,O1,O2,O3,lr)
        W1,b1,W2,b2,W3,b3,vW1,vb1,vW2,vb2,vW3,vb3 = OptimizerMomentum(W1,b1,W2,b2,W3,b3,dW1,dB1,dW2,dB2,dW3,dB3,vW1,vb1,vW2,vb2,vW3,vb3,lr,beta)
    return W1,b1,W2,b2,W3,b3

def OptimizerSGD(W1,b1,W2,b2,W3,b3,dW1,dB1,dW2,dB2,dW3,dB3,lr):
    W1 = W1 - dW1*lr
    b1 = b1 - dB1*lr
    W2 = W2 - dW2*lr
    b2 = b2 - dB2*lr
    W3 = W3 - dW3*lr
    b3 = b3 - dB3*lr
    return W1,b1,W2,b2,W3,b3
        
def SGD(W1,b1,W2,b2,W3,b3,S,Y,batch_size=16,lr=1e-4,epoch=0):
    n_samples = S.shape[0]
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    S,Y = S[idx],Y[idx]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i + batch_size, n_samples)
        s,y =  S[begin:end] , Y[begin:end]
        O1,O2,O3 = Forward(W1,b1,W2,b2,W3,b3,s)
        dW1,dB1,dW2,dB2,dW3,dB3 = Backward(W1,b1,W2,b2,W3,b3,s,y,O1,O2,O3,lr)
        W1,b1,W2,b2,W3,b3 = OptimizerSGD(W1,b1,W2,b2,W3,b3,dW1,dB1,dW2,dB2,dW3,dB3,lr)
    return W1,b1,W2,b2,W3,b3

def OptimizerAdaGrad(W1,b1,W2,b2,W3,b3,dW1,dB1,dW2,dB2,dW3,dB3,sW1,sb1,sW2,sb2,sW3,sb3,lr):
    params = [W1,b1,W2,b2,W3,b3]
    gradient = [dW1,dB1,dW2,dB2,dW3,dB3]
    squared_grad = [sW1,sb1,sW2,sb2,sW3,sb3]
    for i in range(6):
        squared_grad[i] = squared_grad[i] + np.multiply(gradient[i],gradient[i])
        params[i] = params[i] - lr / np.sqrt(squared_grad[i]+ 1e-5) * gradient[i]

    W1,b1,W2,b2,W3,b3 = params
    sW1,sb1,sW2,sb2,sW3,sb3 = squared_grad
    return W1,b1,W2,b2,W3,b3, sW1,sb1,sW2,sb2,sW3,sb3
        
def AdaGrad(W1,b1,W2,b2,W3,b3,S,Y,batch_size=16,lr=1e-4,beta=0.99,epoch=0):
    n_samples = S.shape[0]
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    S,Y = S[idx],Y[idx]
    sW1,sb1,sW2,sb2,sW3,sb3 = np.zeros_like(W1),np.zeros_like(b1),np.zeros_like(W2),np.zeros_like(b2),np.zeros_like(W3),np.zeros_like(b3)
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i + batch_size, n_samples)
        s,y =  S[begin:end] , Y[begin:end]
        O1,O2,O3 = Forward(W1,b1,W2,b2,W3,b3,s)
        dW1,dB1,dW2,dB2,dW3,dB3 = Backward(W1,b1,W2,b2,W3,b3,s,y,O1,O2,O3,lr)
        W1,b1,W2,b2,W3,b3,sW1,sb1,sW2,sb2,sW3,sb3 = OptimizerAdaGrad(W1,b1,W2,b2,W3,b3,dW1,dB1,dW2,dB2,dW3,dB3,sW1,sb1,sW2,sb2,sW3,sb3,lr)
    return W1,b1,W2,b2,W3,b3

def OptimizerRMSProp(W1,b1,W2,b2,W3,b3,dW1,dB1,dW2,dB2,dW3,dB3,sW1,sb1,sW2,sb2,sW3,sb3,lr,gamma):
    params = [W1,b1,W2,b2,W3,b3]
    gradient = [dW1,dB1,dW2,dB2,dW3,dB3]
    squared_grad = [sW1,sb1,sW2,sb2,sW3,sb3]
    for i in range(6):
        squared_grad[i] = gamma * squared_grad[i] + (1-gamma) * np.multiply(gradient[i],gradient[i])
        params[i] = params[i] - lr / np.sqrt(squared_grad[i]+ 1e-6) * gradient[i]

    W1,b1,W2,b2,W3,b3 = params
    sW1,sb1,sW2,sb2,sW3,sb3 = squared_grad
    return W1,b1,W2,b2,W3,b3, sW1,sb1,sW2,sb2,sW3,sb3
        
def RMSProp(W1,b1,W2,b2,W3,b3,S,Y,batch_size=16,lr=1e-4,gamma=0.9,epoch=0):
    n_samples = S.shape[0]
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    S,Y = S[idx],Y[idx]
    sW1,sb1,sW2,sb2,sW3,sb3 = np.zeros_like(W1),np.zeros_like(b1),np.zeros_like(W2),np.zeros_like(b2),np.zeros_like(W3),np.zeros_like(b3)
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i + batch_size, n_samples)
        s,y =  S[begin:end] , Y[begin:end]
        O1,O2,O3 = Forward(W1,b1,W2,b2,W3,b3,s)
        dW1,dB1,dW2,dB2,dW3,dB3 = Backward(W1,b1,W2,b2,W3,b3,s,y,O1,O2,O3,lr)
        W1,b1,W2,b2,W3,b3,sW1,sb1,sW2,sb2,sW3,sb3 = OptimizerRMSProp(W1,b1,W2,b2,W3,b3,dW1,dB1,dW2,dB2,dW3,dB3,sW1,sb1,sW2,sb2,sW3,sb3,lr,gamma)
    return W1,b1,W2,b2,W3,b3

def OptimizerAdam(W1,b1,W2,b2,W3,b3,dW1,dB1,dW2,dB2,dW3,dB3,vW1,vb1,vW2,vb2,vW3,vb3,sW1,sb1,sW2,sb2,sW3,sb3,lr,beta1,beta2,t):
    momentum = [vW1,vb1,vW2,vb2,vW3,vb3]
    second_momen = [sW1,sb1,sW2,sb2,sW3,sb3]
    gradient = [dW1,dB1,dW2,dB2,dW3,dB3]
    params = [W1,b1,W2,b2,W3,b3]
    for i in range(6):
        momentum[i] = beta1 * momentum[i] + (1-beta1) * gradient[i]
        second_momen[i] = beta2 * second_momen[i] + (1-beta2) * np.multiply(gradient[i],gradient[i])
        v_hat = momentum[i] / (1 - beta1**t)
        s_hat = second_momen[i] / (1 - beta2**t)
        g_t = np.multiply(lr * v_hat,1 / (np.sqrt(s_hat)+1e-6))
        params[i] = params[i] - g_t
    W1,b1,W2,b2,W3,b3 = params
    vW1,vb1,vW2,vb2,vW3,vb3 = momentum
    sW1,sb1,sW2,sb2,sW3,sb3 = second_momen
    return W1,b1,W2,b2,W3,b3,vW1,vb1,vW2,vb2,vW3,vb3,sW1,sb1,sW2,sb2,sW3,sb3
        
def Adam(W1,b1,W2,b2,W3,b3,S,Y,batch_size=16,lr=1e-4,beta1=0.9,beta2=0.999,epoch=0):
    n_samples = S.shape[0]
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    S,Y = S[idx],Y[idx]
    vW1,vb1,vW2,vb2,vW3,vb3 = np.zeros_like(W1),np.zeros_like(b1),np.zeros_like(W2),np.zeros_like(b2),np.zeros_like(W3),np.zeros_like(b3)
    sW1,sb1,sW2,sb2,sW3,sb3 = np.zeros_like(W1),np.zeros_like(b1),np.zeros_like(W2),np.zeros_like(b2),np.zeros_like(W3),np.zeros_like(b3)
    t = epoch
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i + batch_size, n_samples)
        s,y =  S[begin:end] , Y[begin:end]
        O1,O2,O3 = Forward(W1,b1,W2,b2,W3,b3,s)
        dW1,dB1,dW2,dB2,dW3,dB3 = Backward(W1,b1,W2,b2,W3,b3,s,y,O1,O2,O3,lr)
        W1,b1,W2,b2,W3,b3,vW1,vb1,vW2,vb2,vW3,vb3,sW1,sb1,sW2,sb2,sW3,sb3 = OptimizerAdam(W1,b1,W2,b2,W3,b3,dW1,dB1,dW2,dB2,dW3,dB3,vW1,vb1,vW2,vb2,vW3,vb3,sW1,sb1,sW2,sb2,sW3,sb3,lr,beta1,beta2,t)
    return W1,b1,W2,b2,W3,b3

def OptimizerYogi(W1,b1,W2,b2,W3,b3,dW1,dB1,dW2,dB2,dW3,dB3,vW1,vb1,vW2,vb2,vW3,vb3,sW1,sb1,sW2,sb2,sW3,sb3,lr,beta1,beta2,t):
    momentum = [vW1,vb1,vW2,vb2,vW3,vb3]
    second_momen = [sW1,sb1,sW2,sb2,sW3,sb3]
    gradient = [dW1,dB1,dW2,dB2,dW3,dB3]
    params = [W1,b1,W2,b2,W3,b3]
    for i in range(6):
        momentum[i] = beta1 * momentum[i] + (1-beta1) * gradient[i]
        g_t2 = np.multiply(gradient[i],gradient[i])
        second_momen[i] = second_momen[i] + (1-beta2) * np.multiply(g_t2,np.sign(g_t2-second_momen[i]))
        v_hat = momentum[i] / (1 - beta1**t)
        s_hat = second_momen[i] / (1 - beta2**t)
        g_t = np.multiply(lr * v_hat,1 / (np.sqrt(s_hat)+1e-6))
        params[i] = params[i] - g_t
    W1,b1,W2,b2,W3,b3 = params
    vW1,vb1,vW2,vb2,vW3,vb3 = momentum
    sW1,sb1,sW2,sb2,sW3,sb3 = second_momen
    return W1,b1,W2,b2,W3,b3,vW1,vb1,vW2,vb2,vW3,vb3,sW1,sb1,sW2,sb2,sW3,sb3
        
def Yogi(W1,b1,W2,b2,W3,b3,S,Y,batch_size=16,lr=1e-4,beta1=0.9,beta2=0.999,epoch=0):
    n_samples = S.shape[0]
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    S,Y = S[idx],Y[idx]
    vW1,vb1,vW2,vb2,vW3,vb3 = np.zeros_like(W1),np.zeros_like(b1),np.zeros_like(W2),np.zeros_like(b2),np.zeros_like(W3),np.zeros_like(b3)
    sW1,sb1,sW2,sb2,sW3,sb3 = np.zeros_like(W1),np.zeros_like(b1),np.zeros_like(W2),np.zeros_like(b2),np.zeros_like(W3),np.zeros_like(b3)
    t = epoch
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i + batch_size, n_samples)
        s,y =  S[begin:end] , Y[begin:end]
        O1,O2,O3 = Forward(W1,b1,W2,b2,W3,b3,s)
        dW1,dB1,dW2,dB2,dW3,dB3 = Backward(W1,b1,W2,b2,W3,b3,s,y,O1,O2,O3,lr)
        W1,b1,W2,b2,W3,b3,vW1,vb1,vW2,vb2,vW3,vb3,sW1,sb1,sW2,sb2,sW3,sb3 = OptimizerYogi(W1,b1,W2,b2,W3,b3,dW1,dB1,dW2,dB2,dW3,dB3,vW1,vb1,vW2,vb2,vW3,vb3,sW1,sb1,sW2,sb2,sW3,sb3,lr,beta1,beta2,t)
    return W1,b1,W2,b2,W3,b3