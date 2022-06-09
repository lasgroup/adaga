"""
Gaussian Process class, with NumPy.

Edoardo Caldarelli, ETH Zurich
Code based on Gabriele Abbati, Machine Learning Research Group, University of Oxford
May 2021
"""



# Libraries
from .kernels import *
import numpy as np
import sys


class GaussianProcess(object):
    """
    Gaussian Process class for regression.
    """

    def __init__(self, input_dim: int, n_points: int, n_feat: int, jitter, variances, lengthscales, noise_vars, kernel: str = 'RBF'):
        """
        Constructor.
        :param input_dim: number of states;
        :param n_points: number of observation points;
        :param lengthscales: the GP kernel lengthscales (1 per system state);
        :param kern_variances: the GP kernel variances (1 per system state);
        :param noise_variances: the GP likelihood variance (1 per system state);
        :param kernel: string indicating which kernel to use for regression.
        Valid option is 'RBF'.
        """
        self.n_states = input_dim
        self.n_feat = n_feat
        self.n_points = n_points
        self.jitter = jitter
        self.kernel = self._initialize_kernel(input_dim, variances=variances, lengthscales=lengthscales, kernel=kernel)
        self.likelihood_variances = noise_vars
        # GP factor matrices
        self.k_mm = None
        self.k_nm = None
        self.diff_k_nm = None
        return

    @staticmethod
    def _initialize_kernel(input_dim: int, variances, lengthscales, kernel: str = 'RBF') -> GenericKernelWithQFF:
        """
        Initialize the kernel of the Gaussian Process.
        :param input_dim: number of states;
        :param kernel: string indicating which kernel to use for regression.
        Valid option is 'RBF'.
        :return: the GenericKernel object.
        """
        if kernel == 'RBF':
            return RBFKernelWithQFF(input_dim, variances, lengthscales)
        else:
            sys.exit("Error: specified Gaussian Process kernel not valid")

    def build_supporting_factor_covariance_matrices(self, t: np.array) -> None:
        """
        Pre-compute the GP factor matrices that appear as constants in ODIN risk.
        :param t: time stamps of the training set;
        """
        self.k_mm = self._build_identity_matrix()
        self.k_nm = self._build_hermite_embedding(t)
        self.diff_k_nm = self._build_diff_hermite_embedding(t)
        self.k_mm_inv_k_mn = np.linalg.solve(self.k_mm, np.transpose(self.k_nm, [0, 2, 1]))
        self.k_mm_inv_k_mn_diff = np.linalg.solve(self.k_mm, np.transpose(self.diff_k_nm, [0, 2, 1]))
        self.a = self.jitter * self.k_mm + np.matmul(np.transpose(self.k_nm, [0, 2, 1]), self.k_nm)
        self.a_inv_k_mn = np.linalg.solve(self.a, np.transpose(self.k_nm, [0, 2, 1]))

        return

    def _build_identity_matrix(self) -> np.array:
        """
        Build the identity matrix.
        :return: the tensors containing the matrix.
        """
        c_phi_matrices = np.expand_dims(np.identity(self.n_feat), axis=0)
        return c_phi_matrices

    def _build_hermite_embedding(self, t: np.array) -> np.array:
        """
        Build the cross covariance matrices between the training data and the
        new test points: K(x_train, x_test).
        :param t: time stamps of the training set;
        :return: the tensors containing the matrices.
        """
        cross_c_phi_matrices = self.kernel.compute_c_phi(t, n_feat=self.n_feat)
        return cross_c_phi_matrices

    def _build_diff_hermite_embedding(self, t: np.array)\
            -> np.array:
        """
        Builds the matrices diff_c_phi: dK(t,t') / dt.
        :param t: time stamps of the training set;
        :return the tensor containing the matrices.
        """
        diff_c_phi_matrices = self.kernel.compute_diff_c_phi(t, n_feat=self.n_feat)
        return diff_c_phi_matrices

    def compute_efficient_prior_inverse_part(self) -> np.array:
        """
        Compute the inverse of the prior covariance matrix via matrix inversion lemma.
        :return: the tensor containing the matrix.
        """
        a_final = np.matmul(self.k_nm, self.a_inv_k_mn)
        return a_final

    def compute_posterior_mean(self, system: np.array) -> np.array:
        """
        Compute the mean of GP the posterior.
        :param system: values of the states of the system;
        :return: the TensorFlow tensor with the mean.
        """
        k_nn = np.matmul(self.k_nm, self.k_mm_inv_k_mn) + self.jitter * np.expand_dims(np.identity(self.n_points), axis=0)

        inner_matrix = (self.likelihood_variances + self.jitter) * self.k_mm + \
                       np.matmul(np.transpose(self.k_nm, [0, 2, 1]), self.k_nm)
        rhs = np.linalg.solve(inner_matrix, np.transpose(self.k_nm, [0, 2, 1]))
        second_term = np.matmul(self.k_nm, rhs)
        k_nn_inv = np.reciprocal(self.likelihood_variances + self.jitter) * (np.identity(self.n_points) - second_term)
        factor = np.matmul(k_nn_inv, np.expand_dims(system, -1))
        mu = np.matmul(k_nn, factor)
        return mu

    def compute_d_matrix(self) -> np.array:
        """
        Compute the D matrix in ODIN risk.
        :return: the TensorFlow tensor with the matrix.
        """
        d_matrix = np.matmul(self.diff_k_nm, self.a_inv_k_mn)
        return d_matrix
