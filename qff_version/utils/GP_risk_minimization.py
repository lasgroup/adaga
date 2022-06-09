"""
Implementation of the Approximate Gaussian process marginal likelihood minimization algorithm.

Emmanouil Angelis, ETH Zurich 
based on code from
Gabriele Abbati, Machine Learning Research Group, University of Oxford
February 2019
"""

# Libraries
from odin.utils.gaussian_processes import GaussianProcess
from odin.utils.tensorflow_optimizer import ExtendedScipyOptimizerInterface
import numpy as np
import tensorflow as tf
from typing import Union, Tuple
import time

class GPRiskMinimization(object):
    """
    Class that implements marginal likelihood minimization for hyperparameter training of GP.
    """

    def __init__(self, system_data: np.array, t_data: np.array,
                 gp_kernel: str = 'RBF',
                 single_gp: bool = False,
                 state_normalization: bool = True,
                 time_normalization: bool = True):
        """
        Constructor
        :param system_data: numpy array containing the noisy observations of the state values of the system, size is [dim, n_points];
        :param t_data: numpy array containing the time stamps corresponding to the observations passed as system_data;
        :param gp_kernel: string indicating which kernel to use in the GP. Valid options are ONLY 'RBF' for the current implementation;
        :param single_gp: boolean, indicates whether to use a single set of GP hyperparameters for each state;
        :param state_normalization: boolean, indicates whether to normalize the states values;
        :param time_normalization: boolean, indicates whether to normalize the time stamps;
        """
        # Save arguments
        self.system_data = np.copy(system_data)
        self.t_data = np.copy(t_data).reshape(-1, 1)
        self.dim, self.n_p = system_data.shape
        self.gp_kernel = gp_kernel
        self.single_gp = single_gp

        # Initialize utils
        self._compute_standardization_data(state_normalization,
                                           time_normalization)
        # TensorFlow placeholders and constants
        self._build_tf_data()
        # Initialize the Gaussian Process for the derivative model
        self.gaussian_process = GaussianProcess(self.dim, self.n_p,
                                                self.gp_kernel, self.single_gp)
        # Initialization of TF operations
        self.init = None
        self.negative_data_loglikelihood = None
        return

    def _compute_standardization_data(self, state_normalization: bool,
                                      time_normalization: bool) -> None:
        """
        Compute the means and the standard deviations for data standardization,
        used in the GP training.
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
        # For the sigmoid kernel the input time values must be positive, i.e.
        # we only divide by the standard deviation
        if self.gp_kernel == 'Sigmoid':
            self.t_data_mean = 0.0
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
        self.n_points = tf.constant(self.n_p, dtype=tf.int32)
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
        gp_kern_lengthscale_bounds = (np.log(1e-6), np.log(100.0))
        gp_kern_variance_bounds = (np.log(1e-6), np.log(100.0))
        gp_kern_likelihood_bounds = (np.log(1e-6), np.log(100.0))
        # Dictionary construction
        var_to_bounds = {gp_vars[0]: gp_kern_lengthscale_bounds,
                         gp_vars[1]: gp_kern_variance_bounds,
                         gp_vars[2]: gp_kern_likelihood_bounds}
        return var_to_bounds

    @staticmethod
    def _build_var_to_bounds_gp_sigmoid() -> dict:
        """
        Builds the dictionary containing the bounds that will be applied to the
        variable in the Gaussian Process model (specific for the sigmoid
        kernel).
        :return: the dictionary variables to bounds.
        """
        # Extract TF variables and select the GP ones
        t_vars = tf.trainable_variables()
        gp_vars = [var for var in t_vars if 'gaussian_process' in var.name]
        # Bounds for the GP hyper-parameters
        gp_kern_a_bounds = (np.log(1e-6), np.log(100.0))
        gp_kern_b_bounds = (np.log(1e-6), np.log(100.0))
        gp_kern_variance_bounds = (np.log(1e-6), np.log(100.0))
        gp_kern_likelihood_bounds = (np.log(1e-6), np.log(100.0))
        # Dictionary construction
        var_to_bounds = {gp_vars[0]: gp_kern_a_bounds,
                         gp_vars[1]: gp_kern_b_bounds,
                         gp_vars[2]: gp_kern_variance_bounds,
                         gp_vars[3]: gp_kern_likelihood_bounds}
        return var_to_bounds

    def _train_data_based_gp(self, session: tf.Session()) -> None:
        """
        Performs a classic GP regression on the data of the system. For each
        state of the system we train a different GP by maximum likelihood to fix
        the kernel hyper-parameters.
        :param session: TensorFlow session used during the optimization.
        """
        # Extract TF variables and select the GP ones
        t_vars = tf.trainable_variables()
        gp_vars = [var for var in t_vars if 'gaussian_process' in var.name]
        # Build the bounds for the GP hyper-parameters
        if self.gp_kernel == 'Sigmoid':
            var_to_bounds = self._build_var_to_bounds_gp_sigmoid()
        else:
            var_to_bounds = self._build_var_to_bounds_gp()
        # Initialize the TF/scipy optimizer
        self.data_gp_optimizer = ExtendedScipyOptimizerInterface(
            self.negative_data_loglikelihood, method="L-BFGS-B",
            var_list=gp_vars, var_to_bounds=var_to_bounds)
        # Optimize
        self.data_gp_optimizer.basinhopping(session, n_iter=50, stepsize=0.05)
        return

    def build_model(self) -> None:
        """
        Builds Some common part of the computational graph for the optimization.
        """
        self.gaussian_process.build_supporting_covariance_matrices(
            self.t, self.t)
        self.negative_data_loglikelihood = \
            - self.gaussian_process.compute_average_log_likelihood(self.system)
        return

    def _initialize_variables(self) -> None:
        """
        Initialize all the variables and placeholders in the graph.
        """
        self.init = tf.global_variables_initializer()
        return

    def train(self) -> [int, np.array, np.array, np.array]:
        """
        Trains the GP, i.e tuning the hyperparameters
        Returns the time needed for the optimization, as well as the hyperpameters found
        """
        self._initialize_variables()
        session = tf.Session()
        with session:
            # Start the session
            session.run(self.init)
            # Train the GP
            secs=time.time()
            self._train_data_based_gp(session)
            secs=time.time() -secs
            print("Likelihood is ",session.run(self.negative_data_loglikelihood))
            # Print GP hyperparameters
            print("GP trained ------------------------------------------------")
            if self.gp_kernel == 'Sigmoid':
                a=session.run(self.gaussian_process.kernel.a)
                b=session.run(self.gaussian_process.kernel.b)
                variances=session.run(self.gaussian_process.diff_kernel.variances)
                likelihood_variances=session.run(self.gaussian_process.likelihood_variances)
                print("a:",a)
                print("b:",b)
                print("variances:",variances)
                res = [secs,a,b,variances,likelihood_variances]
            else:
                lengthscales=session.run(self.gaussian_process.kernel.lengthscales)
                variances=session.run(self.gaussian_process.kernel.variances)
                likelihood_variances=session.run(self.gaussian_process.likelihood_variances)
                print("lengthscales:",lengthscales)
                print("variances:",variances)
                print("likelihood_variances:",likelihood_variances)
                res = [secs,lengthscales,variances,likelihood_variances]
            print("-----------------------------------------------------------")
        tf.reset_default_graph()
        return res
