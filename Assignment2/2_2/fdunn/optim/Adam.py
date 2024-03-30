import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from fdunn.optim.base import Optimizer

class Adam(Optimizer):
    def __init__(self, model, lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.model = model
        self.lr = lr
        self.beta1 = beta_1
        self.beta2 = beta_2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self):
        """
        Performs a single optimization step.
        """
        ###########################################################################
        # TODO:                                                                   #
        # Implement the step method.                                              #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        if self.m is None:
            self.m = []
            self.v = []
            for layer in self.model.layers:
                if isinstance(layer.params, dict):
                    m_dict = {}
                    v_dict = {}
                    for key in layer.params.keys():
                        m_dict[key] = np.zeros_like(layer.params[key])
                        v_dict[key] = np.zeros_like(layer.params[key])
                    self.m.append(m_dict)
                    self.v.append(v_dict)
        
        self.t += 1
        idx = 0
        for i in range(len(self.model.layers)):
            if isinstance(self.model.layers[i].params, dict):
                for key in self.model.layers[i].params.keys():
                    self.m[idx][key] = self.beta1 * self.m[idx][key] + (1 - self.beta1) * self.model.layers[i].grads[key]
                    self.v[idx][key] = self.beta2 * self.v[idx][key] + (1 - self.beta2) * (self.model.layers[i].grads[key]**2)
                    m_hat = self.m[idx][key] / (1 - self.beta1**self.t)
                    v_hat = self.v[idx][key] / (1 - self.beta2**self.t)
                    self.model.layers[i].params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                idx += 1

