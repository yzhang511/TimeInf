"""
Leave Within Structure Out Cross-Validation (LWCV)
"""
import autograd
import autograd.numpy as np


class LWCV:
    def __init__(self, model, theta_one, T, o):
        """
        :param model: Must implement weighted_loss
        :param theta_one: MLE / MAP fit on the entire dataset (with all weights set to 1)
        :param T: length of sequence / number of sites in a MRF
        :param o: Indices of sites in the held out fold
        """
        self.weights_one = np.ones([T])
        self.theta_one = theta_one
        self.o = o
        self.H = self.compute_hessian(model)
        self.J = self.compute_Jmatrix(model)

    def compute_hessian(self, model):
        eval_hess = autograd.hessian(model.weighted_loss, argnum=0)
        H = eval_hess(self.theta_one, self.weights_one)
        return H

    def compute_Jmatrix(self, model):
        eval_d2l_dParams_dWeights = autograd.jacobian(autograd.jacobian(model.weighted_loss, argnum=0), argnum=1)
        J = eval_d2l_dParams_dWeights(self.theta_one, self.weights_one)
        return J

    def compute_params_acv(self):
        """
        Leave within structure cross-validation. N=1.
        :return: Approximated parameter
        """
        HinvJ = -np.linalg.solve(self.H, self.J)
        weights = np.ones_like(self.weights_one)
        weights[self.o] = 0  # dropped sites
        params_acv = self.theta_one + HinvJ.dot(weights - 1)
        return params_acv


