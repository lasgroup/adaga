"""
Edoardo Caldarelli, ETH Zurich
"""

import numpy as np
import gpflow
import tensorflow as tf
from utils.stat_test import StatisticalTest
np.set_printoptions(threshold=np.inf)
from gpflow import features
from gpflow.logdensities import multivariate_normal

class AdaptiveRegionalization(object):
    """
    Class that regionalizes the time domain in a streaming fashion.
    """

    def __init__(self, domain_data,
                 system_data,
                 delta,
                 min_w_size,
                 n_ind_pts,
                 seed,
                 batch_size,
                 kern="RBF",
                 domain_test=None,
                 system_test=None,
                 input_horizon=None):
        """
        Constructor
        :param domain_data: [n x 1] array of timesteps;
        :param system_data: [n x 1] array of observations;
        :param delta: the delta hyperparameter to be used in the thresholds;
        :param min_w_size: the minimum window size allowed;
        :param n_ind_pts: the number of inducing points to use;
        :param seed: the seed (fixed for reproducibility);
        :param n_batches: the number of batches in which the overll trajectory is partitioned;
        :param kern: the kernel to be used.
        """
        self.x = domain_data
        self.y = system_data
        self.n_states, self.n_points = system_data.shape
        self.delta = delta
        self.min_window_size = min_w_size
        self.num_inducing_points = n_ind_pts
        self.batch_time_jump = batch_size
        self.seed = seed
        self.closed_windows = []
        self.kern = kern
        self.domain_test = domain_test
        self.y_test = system_test
        # This param is needed to decouple the minimum window size from a reduced time horizon
        # (w.r.t. the final value of the trajectory).
        self.input_horizon = input_horizon
        if self.input_horizon is not None:
            self._slice_domain_function()

    def _slice_domain_function(self):
        sliced_x_y = np.array([e for e in np.column_stack((self.x, self.y)) if e[0] <= self.input_horizon])
        self.x = np.expand_dims(sliced_x_y[:, 0], axis=-1)
        self.y = np.expand_dims(sliced_x_y[:, 1], axis=-1)


    def _create_expert(self, window, new: bool, x_mean=None, x_std=None, y_mean=None, y_std=None):
        """
        This method creates the expert on the region of interest (full window or overlap);
        :param window: the slice of data that supports the expert;
        :param new: whether the expert is trained on the overlap or not;
        :param x_mean: the mean to use in time standardization;
        :param x_std: the std dev to use in time standardization;
        :param y_mean: the mean to use in observations' standardization;
        :param y_std: the std dev to use in observations' standardization;
        :return: the expert, together with the (potentially recomputed) time and observations' mean, atd dev.
        """
        x = np.expand_dims(window[:, 0], axis=-1)
        y = np.expand_dims(window[:, 1], axis=-1)
        if not new:
            x_mean = np.mean(x)
            x_std = np.std(x)
            y_mean = np.mean(y)
            y_std = np.std(y)
        x -= x_mean
        x /= x_std
        y -= y_mean
        y /= y_std
        z_init = np.random.choice(x[:, 0], min(self.num_inducing_points, x.shape[0]), replace=False)
        z_init = np.expand_dims(z_init, axis=-1)

        with gpflow.defer_build():
            if self.kern == "RBF":
                k = gpflow.kernels.RBF(input_dim=1)
                k.lengthscales.transform = gpflow.transforms.Logistic(1e-3, 100)
            elif self.kern == "Matern52":
                print(self.kern)
                k = gpflow.kernels.Matern52(input_dim=1)
                k.lengthscales.transform = gpflow.transforms.Logistic(1e-3, 100)
            elif self.kern == "RQ":
                print(self.kern)
                k = gpflow.kernels.RationalQuadratic(input_dim=1)
                k.lengthscales.transform = gpflow.transforms.Logistic(1e-3, 100)
            elif self.kern == "Periodic":
                base = gpflow.kernels.RBF(input_dim=1)
                if not new:
                    base.variance.transform = gpflow.transforms.Logistic(1e-8, 1.4)
                base.lengthscales.transform = gpflow.transforms.Logistic(1e-3, 100)
                k = gpflow.kernels.Periodic(base=base)
            elif self.kern == "Linear":
                k = gpflow.kernels.Linear(input_dim=1)

            expert = gpflow.models.SGPR(X=x, Y=y, kern=k, Z=z_init)
        k.variance = 1.0
        if not new and self.kern != "Periodic":
            k.variance.transform = gpflow.transforms.Logistic(1e-8, 1.4)

        expert.compile()
        return expert, x_mean, x_std, y_mean, y_std

    def _build_likelihood(self, model: gpflow.models.SGPR):
        """
        This method builds the likelihood of the given model.
        """
        # gpflow.models.GPR
        K_uf = gpflow.get_default_session().run(features.Kuf(model.feature, model.kern, model.X.value))
        K_uu = gpflow.get_default_session().run(
            features.Kuu(model.feature, model.kern, jitter=gpflow.settings.numerics.jitter_level))
        K_uu_inv = np.linalg.inv(K_uu)
        K = np.matmul(np.matmul(np.transpose(K_uf), K_uu_inv), K_uf) + np.identity(model.X.value.shape[0],
                                                                                   dtype=gpflow.settings.float_type) \
            * model.likelihood.variance.value
        L = np.linalg.cholesky(K)
        m = model.mean_function(model.X.value)
        y_tensor = tf.constant(model.Y.value)
        logpdf = gpflow.get_default_session().run(
            multivariate_normal(y_tensor, m, L))  # (R,) log-likelihoods for each independent dimension of Y

        return np.sum(logpdf)

    def _build_norm_const(self, model_1: gpflow.models.SGPR, model_2: gpflow.models.SGPR):
        """
        This method builds the normalization constant of the product of two Gaussian pdfs.
        """
        # gpflow.models.GPR
        K_uf_1 = gpflow.get_default_session().run(features.Kuf(model_1.feature, model_1.kern, model_1.X.value))
        K_uu_1 = gpflow.get_default_session().run(
            features.Kuu(model_1.feature, model_1.kern, jitter=gpflow.settings.numerics.jitter_level))
        K_uu_inv_1 = np.linalg.inv(K_uu_1)
        K_1 = np.matmul(np.matmul(np.transpose(K_uf_1), K_uu_inv_1), K_uf_1) + np.identity(model_1.X.value.shape[0],
                                                                                           dtype=gpflow.settings.float_type) \
              * model_1.likelihood.variance.value

        K_uf_2 = gpflow.get_default_session().run(features.Kuf(model_2.feature, model_2.kern, model_2.X.value))
        K_uu_2 = gpflow.get_default_session().run(
            features.Kuu(model_2.feature, model_2.kern, jitter=gpflow.settings.numerics.jitter_level))
        K_uu_inv_2 = np.linalg.inv(K_uu_2)
        K_2 = np.matmul(np.matmul(np.transpose(K_uf_2), K_uu_inv_2), K_uf_2) + np.identity(model_2.X.value.shape[0],
                                                                                           dtype=gpflow.settings.float_type) \
              * model_2.likelihood.variance.value

        L = np.linalg.cholesky(K_1 + K_2)
        m_1 = model_1.mean_function(model_1.X.value)
        m_2 = model_2.mean_function(model_2.X.value)
        logpdf = gpflow.get_default_session().run(
            multivariate_normal(m_1, m_2, L))  # (R,) log-likelihoods for each independent dimension of Y

        return np.sum(logpdf)

    def _build_norm_const_new(self, model_1: gpflow.models.SGPR):
        """
        This method builds the normalization constant of a given model.
        """
        # gpflow.models.GPR
        K_uf_1 = gpflow.get_default_session().run(features.Kuf(model_1.feature, model_1.kern, model_1.X.value))
        K_uu_1 = gpflow.get_default_session().run(
            features.Kuu(model_1.feature, model_1.kern, jitter=gpflow.settings.numerics.jitter_level))
        K_uu_inv_1 = np.linalg.inv(K_uu_1)
        K_1 = np.matmul(np.matmul(np.transpose(K_uf_1), K_uu_inv_1), K_uf_1) + np.identity(model_1.X.value.shape[0],
                                                                                           dtype=gpflow.settings.float_type) \
              * model_1.likelihood.variance.value
        _, c = np.linalg.slogdet(a=2 * np.pi * K_1)
        return 0.5 * c

    def test(self):
        final_pred = np.empty((0, 1))
        final_gt = np.empty((0, 1))
        final_time = np.empty((0, 1))
        for region in self.closed_windows:
            window_test = np.array(
                [e for e in np.column_stack((self.domain_test, self.y_test)) if
                 region["window_start"] <= e[0] < region["window_end"]])

            model_test, x_mean_test, x_std_test, y_mean_test, y_std_test = self._create_expert(window_test, False)
            opt_test = gpflow.train.ScipyOptimizer()
            opt_test.minimize(model_test)

            x_pred_test = np.expand_dims(window_test[:, 0], axis=-1)
            pred, _ = model_test.predict_f(x_pred_test)
            pred = pred * y_std_test + y_mean_test
            y_gt = np.expand_dims(window_test[:, 1], axis=-1) * y_std_test + y_mean_test

            final_pred = np.concatenate((final_pred, pred), axis=0)
            final_gt = np.concatenate((final_gt, y_gt), axis=0)
            final_time = np.concatenate((final_time, x_pred_test * x_std_test + x_mean_test), axis=0)
            gpflow.reset_default_graph_and_session()

        gpflow.reset_default_graph_and_session()

        self.rmse = [self.x.shape[0], np.sqrt(np.sum((final_pred - final_gt) ** 2) / final_gt.shape[0])]

    def regionalize(self) -> None:
        """
        This method applies ADAGA streaming GP regression.
        """
        start = self.x[0, 0]
        end = start + 2 * self.min_window_size # + self.batch_time_jump
        close_current_window = False
        new_window = True
        while True:
            gpflow.reset_default_graph_and_session()
            tf.set_random_seed(self.seed)

            window = np.array([e for e in np.column_stack((self.x, self.y)) if start <= e[0] < end])
            print("start, end:", start, end)
            print("WINDOW SHAPE", window.shape)

            if window.shape[0] <= 1:
                break

            best_start_new_exp = end - self.min_window_size

            window_current_expert = np.array([e for e in window if start <= e[0] < end])
            model_current_expert, x_mean, x_std, y_mean, y_std = self._create_expert(window_current_expert, False)
            opt_current_expert = gpflow.train.ScipyOptimizer()
            opt_current_expert.minimize(model_current_expert)

            if min(end, self.x[-1, 0]) - start > self.min_window_size + 3 > end - self.x[-1, 0]:

                window_new_expert = np.array([e for e in window if best_start_new_exp <= e[0] < end])
                model_new_expert, _, _, _, _ = self._create_expert(window_new_expert, True, x_mean, x_std, y_mean, y_std)
                model_current_expert.X = model_new_expert.X.value
                model_current_expert.Y = model_new_expert.Y.value
                opt_new_expert = gpflow.train.ScipyOptimizer()
                opt_new_expert.minimize(model_new_expert)
                print("CURRENT MODEL", model_current_expert.as_pandas_table())

                print("NEW MODEL", model_new_expert.as_pandas_table())
                statistical_test = StatisticalTest(model_current_expert, model_new_expert, self.delta)
                bad_current_window = statistical_test.test(gpflow.get_default_session())

                if bad_current_window:
                    close_current_window = True

            if not close_current_window or new_window:
                new_window = False

            if end > self.x[-1] or close_current_window:
                new_window = True
                end_test = self.x[-1, 0] if end > self.x[-1] else end - self.min_window_size

                self.closed_windows.append(
                    {"window_start": start, "window_end": end_test})

                start = end - self.min_window_size

                if end > self.x[-1]:
                    break
                end = start + 2 * self.min_window_size


            if not close_current_window or end - start < self.min_window_size or new_window:
                end += self.batch_time_jump

            close_current_window = False
        print("PARTITIONING CREATED:", [(e["window_start"], e["window_end"]) for e in self.closed_windows])
        gpflow.reset_default_graph_and_session()

