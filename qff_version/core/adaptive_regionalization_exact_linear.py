"""
Edoardo Caldarelli, ETH Zurich
"""
import numpy as np
import tensorflow as tf
from utils.GP_exact_linear_risk_minimization import GPLinearRiskMinimization
from utils.stat_test_exact_linear import StatisticalTest
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)


class AdaptiveRegionalization(object):
    """
    Class that regionalizes the time domain in a streaming fashion.
    """

    def __init__(self, domain_data,
                 system_data,
                 delta,
                 min_w_size,
                 seed,
                 n_batches=20):
        """
        Constructor
        :param domain_data: [n x 1] array of timesteps;
        :param system_data: [n x 1] array of observations;
        :param delta: the delta hyperparameter to be used in the thresholds;
        :param min_w_size: the minimum window size allowed;
        :param n_feat: the number of Fourier features to use;
        :param seed: the seed (fixed for reproducibility);
        :param n_batches: the number of batches in which the overll trajectory is partitioned;
        :param kern: the kernel to be used.
        """
        self.x = domain_data
        self.y = system_data
        self.n_states, self.n_points = system_data.shape
        self.delta = delta
        self.min_window_size = min_w_size
        self.seed = seed
        self.n_batches = n_batches
        self.batch_time_jump = (self.x[-1, 0] - self.x[0, 0]) / n_batches
        self.closed_windows = []

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
        y = np.expand_dims(window[:, 1], axis=0)

        if not new:
            x_mean = np.mean(x)
            x_std = np.std(x)
            y_mean = np.mean(y)
            y_std = np.std(y)
        x -= x_mean
        x /= x_std
        y -= y_mean
        y /= y_std
        expert = GPLinearRiskMinimization(y, x, state_normalization=False, new_expert=new, gp_kernel="Linear")
        expert.build_model()
        return expert, x_mean, x_std, y_mean, y_std

    def regionalize(self):
        """
        This method applies ADAGA streaming GP regression.
        """
        start = self.x[0, 0]
        end = start + 2 * self.min_window_size # + self.batch_time_jump
        close_current_window = False
        new_window = True
        good_start, good_end = None, None
        while True:
            tf.reset_default_graph()
            session = tf.Session()
            with session:
                tf.set_random_seed(self.seed)

                window = np.array([e for e in np.column_stack((self.x, self.y)) if start <= e[0] < end])
                print("start, end:", start, end)
                print("WINDOW SHAPE", window.shape)

                if window.shape[0] <= 1:
                    break

                best_start_new_exp = end - self.min_window_size

                window_current_expert = np.array([e for e in window if start <= e[0] < end])
                model_current_expert, x_mean, x_std, y_mean, y_std = self._create_expert(window_current_expert, False)
                _, curr_kern_var, curr_lk_var = model_current_expert.train(session)

            tf.reset_default_graph()
            session = tf.Session()
            with session:
                tf.set_random_seed(self.seed)
                if min(end, self.x[-1, 0]) - start > self.min_window_size + 3 > end - self.x[-1, 0]:
                    window_new_expert = np.array([e for e in window if best_start_new_exp <= e[0] < end])
                    model_new_expert, _, _, _, _ = self._create_expert(window_new_expert, True, x_mean, x_std, y_mean, y_std)
                    _, new_kern_var, new_lk_var = model_new_expert.train(session)

                    # Note that the methods for computing the hermite embeddings require sqrt of kern variance and inverse lth.
                    model_current_expert_dict = {'variances': np.sqrt(curr_kern_var),
                                                 'likelihood_variances': curr_lk_var,
                                                 't': model_new_expert.t_data,
                                                 'system_data': model_new_expert.system_data}
                    model_new_expert_dict = {'variances': np.sqrt(new_kern_var),
                                             'likelihood_variances': new_lk_var,
                                             't': model_new_expert.t_data,
                                             'system_data': model_new_expert.system_data}

                    statistical_test = StatisticalTest(model_current_expert_dict, model_new_expert_dict, self.delta)
                    bad_current_window = statistical_test.test(session)

                    if bad_current_window:
                        close_current_window = True

                if not close_current_window or new_window:

                    new_window = False
                    good_start = start

                if end > self.x[-1] or close_current_window:
                    new_window = True
                    end_test = self.x[-1, 0] if end > self.x[-1] else end - self.min_window_size

                    self.closed_windows.append(
                        {"window_start": good_start, "window_end": end_test})

                    start = end - self.min_window_size

                    if end > self.x[-1]:
                        break
                    end = start + 2 * self.min_window_size


                if not close_current_window or end - start < self.min_window_size or new_window:
                    end += self.batch_time_jump

                close_current_window = False

        print("PARTITIONING CREATED:", [(e["window_start"], e["window_end"]) for e in self.closed_windows])
        tf.reset_default_graph()
