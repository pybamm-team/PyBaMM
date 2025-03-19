import numpy as np
import torch
import torch.nn as nn
from torch import autograd
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt


global  _eps
_eps = 1e-7


def newton_raphson(func, x0, threshold=1e-6, in_backward_hood = False):
    """
    x0: initial guess, with shape torch.Size([2])
    """
    error = 9999999.9
    x_now = x0.clone()
    # define g function for Newton-Raphson
    def g(x):
        return func(x) - x
    # iteration
    n_iter = -1
    while error > threshold and n_iter < 1000:
        x_now = x_now.requires_grad_() # add grad track here for the sake of autograd.functional.jacobian
        f_now = g(x_now)
        J = autograd.functional.jacobian(g, x_now)
        f_now = torch.reshape(f_now, (2,1)) 
        x_new = x_now - torch.reshape(torch.linalg.pinv(J)@f_now, (2,)) 
        # detach for memory saving
        x_new = x_new.clone().detach() # detach for memory saving
        # clamp
        x_new[0] = torch.max(torch.tensor([0.0+_eps, x_new[0]]))
        x_new[1] = torch.min(torch.tensor([1.0-_eps, x_new[1]])) # +- 1e-6 is for the sake of torch.log. We don't want log function to blow up at x=0!
        x_now = x_now.clone().detach() # detach for memory saving
        # calculate error
        if torch.abs(x_new[0]-x_now[0]) < torch.abs(x_new[1]-x_now[1]):
            error = torch.abs(x_new[0]-x_now[0])
        else:
            error = torch.abs(x_new[1]-x_now[1])
        # step forward
        x_now = x_new.clone()
        n_iter = n_iter + 1
    if n_iter >= 999:
        print("Warning: Max iteration in Newton-Raphson solver reached.")
    return x_now



class FixedPointOperation(nn.Module):
    def __init__(self, G, params_list, T = 300):
        """
        The fixed point operation used in the backward pass of common tangent approach. 
        Write the forward(self, x) function in such weird way so that it is differentiable
        G is the Gibbs free energy function 
        params_list contains the RK params and G0, in the sequence of [Omega0, Omega1, ..., G0]
        """
        super(FixedPointOperation, self).__init__()
        self.G = G
        self.params_list = params_list
        self.T = torch.tensor([T])
    def forward(self, x):
        """x[0] is the left limit of phase coexisting region, x[1] is the right limit"""
        x_alpha = x[0]
        x_beta = x[1]
        g_right = self.G(x_beta, self.params_list, self.T) 
        g_left = self.G(x_alpha, self.params_list, self.T)
        mu_right = autograd.grad(outputs=g_right, inputs=x_beta, create_graph=True)[0]
        mu_left = autograd.grad(outputs=g_left, inputs=x_alpha, create_graph=True)[0]
        x_alpha_new = x_beta - (g_right - g_left)/(mu_left + _eps)
        x_alpha_new = torch.clamp(x_alpha_new , min=0.0+_eps, max=1.0-_eps) # clamp
        x_alpha_new = x_alpha_new.reshape(1)
        x_beta_new = x_alpha + (g_right - g_left)/(mu_right + _eps)
        x_beta_new = torch.clamp(x_beta_new , min=0.0+_eps, max=1.0-_eps) # clamp
        x_beta_new = x_beta_new.reshape(1)
        return torch.cat((x_alpha_new, x_beta_new))


# In case that the above implementation doesn't work
class FixedPointOperationForwardPass(nn.Module):
    def __init__(self, G, params_list, T = 300):
        """
        The fixed point operation used in the forward pass of common tangent approach
        Here we don't use the above implementation (instead we use Pinwen's implementation in Jax-TherMo) to guarantee that the solution converges to the correct places in forward pass
        G is the Gibbs free energy function 
        params_list contains the RK params and G0, in the sequence of [Omega0, Omega1, ..., G0]
        """
        super(FixedPointOperationForwardPass, self).__init__()
        self.G = G
        self.params_list = params_list
        self.T = torch.tensor([T])
    def forward(self, x):
        """x[0] is the left limit of phase coexisting region, x[1] is the right limit"""
        # x_alpha_0 = (x[0]).reshape(1)
        # x_beta_0 = (x[1]).reshape(1)
        x_alpha_now = x[0]
        x_beta_now = x[1]
        x_alpha_now = x_alpha_now.reshape(1)
        x_beta_now = x_beta_now.reshape(1)
        g_left = self.G(x_alpha_now, self.params_list, self.T)
        g_right = self.G(x_beta_now, self.params_list, self.T)
        common_tangent = (g_left - g_right)/(x_alpha_now - x_beta_now)
        dcommon_tangent = 9999999.9
        n_iter_ct = 0
        while dcommon_tangent>1e-4 and n_iter_ct < 300:  
            """
            eq1 & eq2: we want dG/dx evaluated at x1 (and x2) to be the same as common_tangent, i.e. mu(x=x1 or x2) = common_tangent
            then applying Newton-Rapson iteration to solve f(x) = mu(x) - common_tangent, where mu(x) = dG/dx
            Newton-Rapson iteration: x1 = x0 - f(x0)/f'(x0)
            """  
            def eq(x):
                y = self.G(x, self.params_list, self.T) - common_tangent*x
                return y
            # update x_alpha
            dx = torch.tensor(999999.0)
            n_iter_dxalpha = 0.0
            while torch.abs(dx) > 1e-6 and n_iter_dxalpha < 300:
                x_alpha_now = x_alpha_now.requires_grad_()
                value_now = eq(x_alpha_now)
                f_now = autograd.grad(value_now, x_alpha_now, create_graph=True)[0]
                f_prime_now = autograd.grad(f_now, x_alpha_now, create_graph=True)[0]
                dx = -f_now/(f_prime_now)
                x_alpha_now = x_alpha_now + dx
                x_alpha_now = x_alpha_now.clone().detach()
                # clamp
                x_alpha_now = torch.max(torch.tensor([0.0+_eps, x_alpha_now]))
                x_alpha_now = torch.min(torch.tensor([1.0-_eps, x_alpha_now])) 
                x_alpha_now = x_alpha_now.reshape(1)
                n_iter_dxalpha = n_iter_dxalpha + 1
            # update x_beta
            dx = torch.tensor(999999.0)
            n_iter_dxbeta = 0.0
            while torch.abs(dx) > 1e-6 and n_iter_dxbeta < 300:
                x_beta_now = x_beta_now.requires_grad_()
                value_now = eq(x_beta_now)
                f_now = autograd.grad(value_now, x_beta_now, create_graph=True)[0]
                f_prime_now = autograd.grad(f_now, x_beta_now, create_graph=True)[0]
                dx = -f_now/(f_prime_now)
                x_beta_now = x_beta_now + dx
                x_beta_now = x_beta_now.clone().detach()
                # clamp
                x_beta_now = torch.max(torch.tensor([0.0+_eps, x_beta_now]))
                x_beta_now = torch.min(torch.tensor([1.0-_eps, x_beta_now])) 
                x_beta_now = x_beta_now.reshape(1)
                n_iter_dxbeta = n_iter_dxbeta + 1
            # after getting new x1 and x2, calculates the new common tangent, the same process goes on until the solution is self-consistent
            common_tangent_new = (self.G(x_alpha_now, self.params_list, self.T) - self.G(x_beta_now, self.params_list, self.T))/(x_alpha_now - x_beta_now)
            dcommon_tangent = torch.abs(common_tangent_new-common_tangent)
            common_tangent = common_tangent_new.clone().detach()
            n_iter_ct = n_iter_ct + 1
        return torch.cat((x_alpha_now, x_beta_now))

