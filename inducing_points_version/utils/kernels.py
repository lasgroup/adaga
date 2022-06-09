"""
Collection of kernel classes to be used in the Gaussian Process Regression.

Edoardo Caldarelli, ETH Zurich
Code based on Gabriele Abbati, Machine Learning Research Group, University of Oxford
"""


# Libraries
import numpy as np
from abc import ABC, abstractmethod


class GenericKernel(ABC):
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

    @staticmethod
    def _compute_squared_distances(xx: np.array,
                                   yy: np.array) -> np.array:
        """
        Compute the matrices of the squared distances between the tensors xx
        and yy.
                    squared_distances[0, i, j] = || x[i] - y[j] ||**2
        The shape of the returned tensor is [1, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the tensor containing the squared distances.
        """
        xx = np.expand_dims(xx, axis=-1)
        r_xx = xx * xx
        yy = np.expand_dims(yy, axis=-1)
        yy = np.transpose(yy, [0, 2, 1])
        r_yy = yy * yy
        r_xy = np.matmul(xx, yy)
        squared_distances = r_xx - 2.0 * r_xy + r_yy
        return squared_distances

    @abstractmethod
    def compute_c_phi(self, xx: np.array,
                      yy: np.array) -> np.array:
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
    def compute_diff_c_phi(self, xx: np.array,
                           yy: np.array) -> np.array:
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

    def compute_c_phi_diff(self, xx: np.array,
                           yy: np.array) -> np.array:
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
        return - self.compute_diff_c_phi(xx, yy)

class RBFKernel(GenericKernel):
    """
    Implementation of the Radial Basis Function kernel.
    """

    def compute_c_phi(self, xx: np.array,
                      yy: np.array) -> np.array:
        """
        Compute the kernel covariance matrix between xx and yy for each state:
                    c_phi[n_s, i, j] =
                        {var * exp( || x[i] - y[j] ||**2 / (2 * l**2))}_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: the tensor containing the covariance matrices.
        """
        squared_distances = self._compute_squared_distances(xx, yy)
        scaled_distances = - squared_distances /\
            np.power(self.lengthscales, 2.0) * 0.5
        cov_matrix = self.variances * np.exp(scaled_distances)
        return cov_matrix

    def compute_diff_c_phi(self, xx: np.array,
                           yy: np.array) -> np.array:
        """
        Compute the derivative of the kernel covariance matrix between xx and
        yy with respect to xx, for each state:
                    diff_c_phi[n_s, i, j] = d/dxx kernel(xx[i], yy[j])_{n_s}
        The shape of the returned tensor is [n_states, n_points, n_points]
        :param xx: input tensor;
        :param yy: input tensor;
        :return: tensor containing the derivatives of the covariance matrices
        w.r.t. xx.
        """

        cov_matrix = self.compute_c_phi(xx, yy)
        xx = np.expand_dims(xx, -1)
        yy = np.expand_dims(yy, -1)
        distances = xx - np.transpose(yy, [0, 2, 1])
        return - distances / np.power(self.lengthscales, 2.0) * cov_matrix
