"""
Collection of kernel classes to be used in the Gaussian Process Regression.

Edoardo Caldarelli, ETH Zurich
Code based on Gabriele Abbati, Machine Learning Research Group, University of Oxford
"""


# Libraries
import numpy as np
from abc import ABC, abstractmethod


class GenericKernelWithQFF(ABC):
    """
    Generic class for a Gaussian Process kernel.
    """

    def __init__(self, input_dim: int, variances, lengthscales):
        """
        Constructor.
        :param input_dim: number of states.
        :param use_single_gp: boolean, indicates whether to use a single set of
        hyperparameters for all states (useful for extremely scarce data
        setting).
        """
        self.dimensionality = input_dim
        self.variances = variances
        self.lengthscales = lengthscales
        return

    @abstractmethod
    def compute_c_phi(self, xx: np.array, n_feat=50) -> np.array:
        """
        To be implemented, compute the kernel covariance matrix between xx and
        yy for each state:
                    c_phi[n_s, i, j] = kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the tensor containing the covariance matrices.
        """
        return 0

    @abstractmethod
    def compute_diff_c_phi(self, xx: np.array, n_feat=50) -> np.array:
        """
        To be implemented, compute the derivative of the kernel covariance
        matrix between xx and yy with respect to xx, for each state:
                    diff_c_phi[n_s, i, j] = d/dxx kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: tensor containing the derivatives of the covariance matrices
        w.r.t. xx.
        """
        return 0

    def compute_c_phi_diff(self, xx: np.array, n_feat=50) -> np.array:
        """
        To be implemented, compute the derivative of the kernel covariance
        matrix between xx and yy with respect to xx, for each state:
                    c_phi_diff[n_s, i, j] = d/dyy kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        Note: for stationary kernels this is just the negative diff_c_phi
        matrix. Non-stationary kernels should override this method too.
        :param xx: input tensor;
        :param yy: input tensor;
        :return: tensor containing the derivatives of the covariance matrices
        w.r.t. yy.
        """
        return - self.compute_diff_c_phi(xx, n_feat)


class RBFKernelWithQFF(GenericKernelWithQFF):
    """
    Implementation of the Radial Basis Function kernel.
    """

    def _hermite_embeding(self, m, gamma, X):
        """
        Returns a numpy array of dimensions (n_states,n_points,m) that has as rows the QFFs for the RBF Kernel
        :param m: int, m/2 is the order of the Quadrature Scheme (note that QFF vectors are of length m, as we use both sines and cosines);
        :param gamma: numpy array of dimensions (n_states,1,1) containing the lengthscales;
        :param X: numpy array of dimensions (n_points,1) with the time points
        """
        (nodes, weights) = np.polynomial.hermite.hermgauss(m // 2)
        nodes = np.reshape(np.sqrt(2) * nodes, [1, 1, -1]) / gamma
        X = np.reshape(X, [1, -1, 1])
        nodes = nodes * X
        weights = np.sqrt(np.reshape(weights / np.sqrt([np.pi]), [1, 1, -1]))
        cos_nodes = weights * np.cos(nodes)
        sin_nodes = weights * np.sin(nodes)
        return np.concatenate([cos_nodes, sin_nodes], axis=2)

    def _hermite_embeding_derivative(self, m, gamma, X):
        """
        Returns a numpy array of dimensions (n_states,n_points,m) that has as rows the derivatives of QFFs for the RBF Kernel
        :param m: int, m/2 the order of the Quadrature Scheme (note that QFF vectors are of length m, as we use both sines and cosines);
        :param gamma: numpy array of dimensions (n_states,1,1) containing the lengthscales;
        :param X: numpy array of dimensions (n_points,1) with the time points
        """
        (nodes, weights) = np.polynomial.hermite.hermgauss(m // 2)
        nodes = np.reshape(np.sqrt(2) * nodes, [1, 1, -1]) / gamma
        weights = np.sqrt(np.reshape(weights / np.sqrt([np.pi]), [1, 1, -1])) * nodes
        X = np.reshape(X, [1, -1, 1])
        nodes = nodes * X
        cos_nodes = -weights * np.sin(nodes)
        sin_nodes = weights * np.cos(nodes)
        return np.concatenate([cos_nodes, sin_nodes], axis=2)

    def compute_c_phi(self, xx: np.array, n_feat=50) -> np.array:
        """
        Compute the embedding for timesteps in t.
        """
        cov_matrix = np.sqrt(self.variances) * self._hermite_embeding(n_feat, self.lengthscales, xx)
        return cov_matrix

    def compute_diff_c_phi(self, xx: np.array, n_feat=50) -> np.array:
        """
        Compute the derivative embedding for timesteps in t.
        """
        der_cov = np.sqrt(self.variances) * self._hermite_embeding_derivative(n_feat, self.lengthscales, xx)
        return der_cov
