import math
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a


def relu(x):
    a = []
    for item in x:
        a.append(max(0,item))
    return a

x = np.arange(-5., 3., 0.1)
sig = sigmoid(x)
tanh = np.tanh(x)
relu = relu(x)
#plt.plot(x,sig, label='sigmoid')
#plt.plot(x,tanh, label='tanh')
#pylab.legend(loc='upper left')
#plt.plot(x,sig,tanh)
plt.plot(x,sig)
plt.plot(x,tanh)
plt.plot(x,relu)
plt.text(-4.6, 1.8, r'$softmax(\vec{x})=\frac{e^{x_i}}{\Sigma^K_{k=1}e^{x_k}}$ for $i=1,\dots,K$', fontsize=10)
plt.xlabel('Input (output of neuron)')
plt.ylabel('Activation')
plt.gca().legend((r'$sigmoid(x)=\frac{1}{1+e^{-x}}$',r'$tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$', r'$ReLU(x)=max(0,x)$'))
plt.show()