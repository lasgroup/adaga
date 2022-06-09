"""
Implementation of the Approximate Gaussian process marginal likelihood minimization algorithm.

Edoardo Caldarelli, ETH Zurich
Code based on Emmanouil Angelis, ETH Zurich, and Gabriele Abbati, Machine Learning Research Group, University of Oxford
"""


# Libraries
from utils.tensorflow_optimizer import ExtendedScipyOptimizerInterface
import numpy as np
import tensorflow as tf
from typing import Union, Tuple
import time

class GPLinearRiskMinimization(object):
    """
    Class that implements Approximate (i.e QFFs are used) marginal likelihood minimization for hyperparameter training of GP.
    """

    def __init__(self, system_data: np.array, t_data: np.array, new_expert: bool,
                 gp_kernel: str = 'Linear',
                 single_gp: bool = False,
                 state_normalization: bool = True,
                 time_normalization: bool = False):
        """
        Constructor
        :param system_data: numpy array containing the noisy observations of the state values of the system, size is [n_states, n_points];
        :param t_data: numpy array containing the time stamps corresponding to the observations passed as system_data;
        :param gp_kernel: string indicating which kernel to use in the GP. Valid options are ONLY 'RBF' for the current implementation;
        :param single_gp: boolean, indicates whether to use a single set of GP hyperparameters for each state;
        :param state_normalization: boolean, indicates whether to normalize the states values;
        :param time_normalization: boolean, indicates whether to normalize the time stamps;
        :param QFF_approx: int, the order of the quadrature scheme
        """
        # Save arguments
        self.new_expert = new_expert
        self.system_data = np.copy(system_data)
        self.t_data = np.copy(t_data).reshape(-1, 1)
        self.dim, self.n_p = system_data.shape
        self.gp_kernel = gp_kernel
        if self.gp_kernel != 'Linear':
            raise NotImplementedError("Class works with linear kernel only")

        self.single_gp = single_gp

        # Compute the data for the standardization (means and standard deviations)
        self._compute_standardization_data(state_normalization,
                                           time_normalization)
        # Build the necessary TensorFlow tensors
        self._build_tf_data()

        # Initialization of TF operations
        self.init = None
        self.negative_data_loglikelihood = None
        return

    def _compute_standardization_data(self, state_normalization: bool,
                                      time_normalization: bool) -> None:
        """
        Compute the means and the standard deviations for data standardization,
        used in the GP hyperparameter training.
        """
        # Compute mean and std dev of the state and time values
        if state_normalization:
            self.system_data_means = np.mean(self.system_data,
                                             axis=1).reshape(self.dim, 1)
            self.system_data_std_dev = np.std(self.system_data,
                                              axis=1).reshape(self.dim, 1)
        else:
            self.system_data_means = np.zeros([self.dim, 1])
            self.system_data_std_dev = np.ones([self.dim, 1])
        if time_normalization:
            self.t_data_mean = np.mean(self.t_data)
            self.t_data_std_dev = np.std(self.t_data)
        else:
            self.t_data_mean = 0.0
            self.t_data_std_dev = 1.0
        # Normalize states and time
        self.normalized_states = (self.system_data - self.system_data_means) / \
            self.system_data_std_dev
        self.normalized_t_data = (self.t_data - self.t_data_mean) / \
            self.t_data_std_dev
        return

    def _build_tf_data(self) -> None:
        """
        Initialize all the TensorFlow constants needed by the pipeline.
        """
        self.system = tf.constant(self.normalized_states, dtype=tf.float64)
        self.t = tf.constant(self.normalized_t_data, dtype=tf.float64)
        self.system_means = tf.constant(self.system_data_means,
                                        dtype=tf.float64,
                                        shape=[self.dim, 1])
        self.system_std_dev = tf.constant(self.system_data_std_dev,
                                          dtype=tf.float64,
                                          shape=[self.dim, 1])
        self.t_mean = tf.constant(self.t_data_mean, dtype=tf.float64)
        self.t_std_dev = tf.constant(self.t_data_std_dev, dtype=tf.float64)
        self.n_points = tf.constant(self.n_p, dtype=tf.float64)
        self.dimensionality = tf.constant(self.dim, dtype=tf.int32)
        return

    @staticmethod
    def _build_var_to_bounds_gp() -> dict:
        """
        Builds the dictionary containing the bounds that will be applied to the
        variable in the Gaussian Process model.
        :return: the dictionary variables to bounds.
        """
        # Extract TF variables and select the GP ones
        t_vars = tf.trainable_variables()
        gp_vars = [var for var in t_vars if 'gaussian_process' in var.name]
        # Bounds for the GP hyper-parameters
        gp_kern_variance_bounds = (None, None)
        gp_kern_likelihood_bounds = (None, None)
        # Dictionary construction
        var_to_bounds = {gp_vars[0]: gp_kern_variance_bounds,
                         gp_vars[1]: gp_kern_likelihood_bounds}
        return var_to_bounds

    def _train_data_based_gp(self, session: tf.Session()) -> None:
        """
        Performs the GP regression on the data of the system. For each state
        of the system we train a different GP by maximum likelihood to tune
        the kernel hyper-parameters.
        :param session: TensorFlow session used during the optimization.
        """
        # Extract TF variables and select the GP ones
        t_vars = tf.trainable_variables()
        gp_vars = [var for var in t_vars if f'gaussian_process_new_{self.new_expert}' in var.name]
        # Build the bounds for the GP hyper-parameters
        var_to_bounds = self._build_var_to_bounds_gp()
        # Initialize the TF/scipy optimizer
        self.data_gp_optimizer = ExtendedScipyOptimizerInterface(
            self.negative_data_loglikelihood, method="L-BFGS-B",
            var_list=gp_vars)
        # Optimize
        self.data_gp_optimizer.basinhopping(session, n_iter=50, stepsize=0.05)
        return

    def build_model(self) -> None:

        """
        Builds Some common part of the computational graph for the optimization.
        """
        # Gaussian Process Interpolation
        a_lth = 1e-2
        b_lth = 1e2
        a_vars = 1e-1
        if not self.new_expert:
            b_vars = 1.4
        else:
            b_vars = 1e3
        a_lk_var = 1e-8
        b_lk_var = b_vars

        with tf.variable_scope(f'gaussian_process_new_{self.new_expert}_kernel'):
            if self.single_gp:
                self.log_variance = tf.Variable(-3,
                                                dtype=tf.float64,
                                                trainable=True,
                                                name='log_variance')
                self.variances = (a_vars + (b_vars - a_vars) /
                                  (1 + tf.exp(-self.log_variance))) \
                                 * tf.ones([self.dimensionality, 1, 1], dtype=tf.float64)
                self.likelihood_logvariance = tf.Variable(
                    -3, dtype=tf.float64, trainable=True,
                    name='variance_loglik')
                self.likelihood_logvariances = \
                    self.likelihood_logvariance * tf.ones([self.dimensionality,
                                                           1, 1],
                                                          dtype=tf.float64)
            else:

                self.log_variances = tf.Variable(
                    -3 * tf.ones([self.dimensionality, 1, 1],
                                 dtype=tf.float64),
                    dtype=tf.float64, trainable=True, name='variances')
                self.variances = (a_vars + (b_vars - a_vars) /
                                  (1 + tf.exp(-self.log_variances)))

                self.likelihood_logvariances = tf.Variable(
                    -3 * tf.ones([self.dimensionality, 1, 1],
                                 dtype=tf.float64),
                    dtype=tf.float64, trainable=True,
                    name='variances_loglik')
            self.likelihood_variances = (a_lk_var + (b_lk_var - a_lk_var) /
                                         (1 + tf.exp(-self.likelihood_logvariances)))
        Z = self.variances * self.t

        Z_t_y = tf.matmul(Z, tf.expand_dims(self.system, -1), transpose_a=True, name='Z_t_y')
        Kernel_inner_dim = tf.matmul(Z, Z, transpose_a=True, name='Kernel_inner_dim') + (
                    self.likelihood_variances + 1e-3) * tf.eye(self.t_data.shape[1], dtype=tf.float64)
        inv_Z_t_y = tf.linalg.solve(Kernel_inner_dim, Z_t_y, name='inv_Z_t_y')

        a_vector = tf.matmul(Z_t_y, inv_Z_t_y, transpose_a=True, name='reg_risk_main_term')
        first_term = tf.reduce_sum(
            tf.reduce_sum(self.system * self.system, axis=1) / tf.squeeze(self.likelihood_variances)) - tf.reduce_sum(
            a_vector / self.likelihood_variances)

        second_term = tf.reduce_sum(tf.linalg.logdet(Kernel_inner_dim)) + (
                    self.n_points - self.t_data.shape[1]) * tf.reduce_sum(tf.log(self.likelihood_variances))

        self.negative_data_loglikelihood = (0.5 * first_term + 0.5 * second_term) / self.n_points
        return

    def _initialize_variables(self) -> None:
        """
        Initialize all the variables and placeholders in the graph.
        """
        self.init = tf.global_variables_initializer()
        return

    def train(self, session) -> [int, np.array, np.array, np.array]:
        """
        Trains the GP, i.e tuning the hyperparameters
        Returns the time needed for the optimization, as well as the hyperpameters found
        """
        self._initialize_variables()
        # Start the session
        session.run(self.init)
        # Train the GP
        secs=time.time()
        self._train_data_based_gp(session)
        secs=time.time() -secs
        print("Likelihood is ",session.run(self.negative_data_loglikelihood))
        # Print GP hyperparameters
        print("GP trained ------------------------------------------------")
        variances=session.run(self.variances)**2
        likelihood_variances=session.run(self.likelihood_variances)
        print("variances:",variances)
        print("likelihood_variances:",likelihood_variances)
        res = [secs,variances,likelihood_variances]
        print("-----------------------------------------------------------")
        return res
