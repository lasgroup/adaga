"""
Edoardo Caldarelli, ETH Zurich
"""

import numpy as np
from utils.GP_approx_risk_minimization import hermite_embeding
import tensorflow as tf
import scipy.stats
import scipy
np.set_printoptions(threshold=np.inf)


class StatisticalTest(object):
    """
    Class that implements the statistical test, used for detecting window degeneracy.
    """

    def __init__(self, model_current_expert,
                 model_new_expert,
                 delta,
                 m):
        """
        Constructor.
        :param model_current_expert: the model trained on the whole window;
        :param model_new_expert: the model trained on the overlap;
        :param delta: the delta value to use in the thresholds.
        """
        self.model_current_expert = model_current_expert
        self.model_new_expert = model_new_expert
        self.delta = delta
        self.n_feat = m

    def _compute_cov_alt(self, model_0, model_1) -> tf.Tensor:
        """
        Compute the covariance matrix under the alternative hypothesis;
        :param model_0: the model trained on the window;
        :param model_1: the model trained on the overlap;
        :return: the covariance matrix under the alternative hypothesis.
        """
        k_uf = model_0['variances'][0, :, :] * tf.transpose(hermite_embeding(self.n_feat, model_0['gamma'], model_0['t'])[0, :, :])

        h_uf = model_1['variances'][0, :, :] * tf.transpose(hermite_embeding(self.n_feat, model_1['gamma'], model_1['t'])[0, :, :])

        sigma_sq = model_0['likelihood_variances'][0, :, :]
        xi_sq = model_1['likelihood_variances'][0, :, :]
        alpha = 1 / sigma_sq + 1 / xi_sq

        d = alpha * sigma_sq * tf.eye(self.n_feat, dtype=tf.float64) + alpha * tf.matmul(k_uf, k_uf, transpose_b=True) - (1 / sigma_sq) * tf.matmul(k_uf,
                                                                                                                   k_uf,
                                                                                                                   transpose_b=True)
        print("d", d)
        d_inv_k_uf = tf.linalg.solve(d, k_uf)
        a_inv = 1.0 / alpha * (tf.eye(model_0['t'].shape[0], dtype=tf.float64) + (1 / sigma_sq) * tf.matmul(k_uf,
                                                                                                                   d_inv_k_uf,
                                                                                                                   transpose_a=True))

        c = xi_sq * tf.eye(self.n_feat, dtype=tf.float64) + tf.matmul(h_uf, h_uf, transpose_b=True) - (1 / xi_sq) * tf.matmul(h_uf,
                                                                                             tf.matmul(a_inv, h_uf,
                                                                                                       transpose_b=True))
        c_inv_h_mn_a_inv = tf.linalg.solve(c, tf.matmul(h_uf, a_inv))
        final_matrix = (1 / xi_sq) * tf.matmul(a_inv, tf.matmul(h_uf, c_inv_h_mn_a_inv, transpose_a=True))
        result = a_inv + final_matrix
        return result

    def _compute_single_expert_inv_covariance(self, model) -> tf.Tensor:
        """
        Computes the inverse covariance matrix for a given GP model.
        :param model: the model whose inverse covariance matrix is to be computed;
        :return: the inverse covariance matrix.
        """
        k_uf = model['variances'][0, :, :] * tf.transpose(hermite_embeding(self.n_feat, model['gamma'], model['t'])[0, :, :])
        variance = model['likelihood_variances'][0, :, :]
        sigma = tf.sqrt(model['likelihood_variances'][0, :, :])
        A = k_uf / sigma
        AAt = tf.matmul(A, A, transpose_b=True)
        B = AAt + tf.eye(self.n_feat, dtype=tf.float64)
        identity = tf.eye(model['t'].shape[0], dtype=tf.float64) / variance
        Lb = tf.cholesky(B)
        print("A", A)
        c = tf.matrix_triangular_solve(Lb, A, lower=True) / sigma
        matrix = tf.matmul(tf.transpose(c), c)

        return tf.subtract(identity, matrix)

    def _compute_single_expert_covariance(self, model) -> tf.Tensor:
        """
        Computes the covariance matrix for a given GP model.
        :param model: the model whose covariance matrix is to be computed;
        :return: the covariance matrix.
        """
        k_uf = model['variances'][0, :, :] * tf.transpose(hermite_embeding(self.n_feat, model['gamma'], model['t'])[0, :, :])
        matrix = tf.matmul(tf.transpose(k_uf), k_uf)

        variance = model['likelihood_variances'][0, :, :]
        identity = variance * tf.eye(model['t'].shape[0], dtype=tf.float64)

        return tf.add(identity, matrix)

    def _compute_ratio(self, inverse_cov_new_exp, vector) -> tf.Tensor:
        """
        Computes the test statistic.
        :param inverse_cov_new_exp: the inverse covariance matrix of the expert trained on the overlap;
        :param vector: the vector containing the observations;
        :return: the statistic value.
        """
        likelihood = -tf.matmul(tf.matmul(tf.transpose(vector), inverse_cov_new_exp), vector)
        return likelihood

    def _compute_thresholds(self, session, cov_null, cov_alt, cov_new_exp, inverse_cov_new_exp, alt=False):
        """
        Computes the threshold for controlling type I and type II errors.
        :param session: tensorflow session;
        :param cov_null: the covariance matrix under the null hypothesis;
        :param cov_alt: the covariance matrix under the alternative hypothesis;
        :param cov_new_exp: the covariance matrix of the expert trained on the overlap;
        :param inverse_cov_new_exp: the inverse covariance matrix of the model trained on the overlap.
        :param alt: wheteher the threshold is for type II errors, or not;
        :return: the threshold value.
        """
        n = tf.shape(cov_new_exp)[0]
        print(cov_alt, inverse_cov_new_exp)
        if alt:
            matrix = tf.matmul(cov_alt, inverse_cov_new_exp)
        else:
            matrix = tf.matmul(cov_null, inverse_cov_new_exp)

        trace = tf.trace(matrix)
        bound = -trace
        squared_l2_norm_eigvals = tf.trace(tf.matmul(matrix, matrix))
        supp_seed = 0
        tolerance = None
        while True:
            try:
                if tolerance == None:
                    approx_infty_norm = scipy.sparse.linalg.eigs(session.run(matrix), k=1, which="LM", return_eigenvectors=False)
                else:
                    approx_infty_norm = scipy.sparse.linalg.eigs(session.run(matrix), k=1, which="LM", return_eigenvectors=False, tol=tolerance)

                break
            except:
                if tolerance == None:
                    tolerance = 1e-15
                else:
                    tolerance *= 10
                if supp_seed > 15:
                    print("!FAILED CONVERGENCE OF LARG. EIGVAL! Too many times.")
                    exit(1)
                else:
                    print(f"!FAILED CONVERGENCE OF LARG. EIGVAL! Reinitializing seed and increasing tolerance to {tolerance}.")
                    np.random.seed(supp_seed)
                    supp_seed += 1
        nu_squared = tf.cast(4.0 * squared_l2_norm_eigvals, tf.float64)
        alpha = tf.cast(4.0 * approx_infty_norm, tf.float64)
        delta = self.delta
        log_term = 2.0 * tf. \
            cast(tf.log(1.0 / delta), tf.float64)

        v = tf.maximum(tf.sqrt(log_term * nu_squared), log_term * alpha)
        const = v
        if alt:
            bound -= const
        else:
            bound += const

        return session.run(bound)

    def test(self, session):
        """
        Method that tests if the current window is spoiled or not.
        :param session: tensorflow session.
        :return: the result of the test (boolean).
        """
        product_covariance_matrix_null = self._compute_single_expert_covariance(self.model_current_expert)

        new_cov = self._compute_single_expert_covariance(self.model_new_expert)
        inv_new_cov = self._compute_single_expert_inv_covariance(self.model_new_expert)
        product_covariance_matrix_alt = self._compute_cov_alt(self.model_current_expert, self.model_new_expert)

        threshold_null = self._compute_thresholds(session, product_covariance_matrix_null, product_covariance_matrix_alt, new_cov, inv_new_cov)
        threshold_alt = self._compute_thresholds(session, product_covariance_matrix_null, product_covariance_matrix_alt, new_cov, inv_new_cov, alt=True)
        ratio = session.run(self._compute_ratio(inv_new_cov, np.transpose(self.model_new_expert['system_data'])))

        print("Threshold for type I errors:", threshold_null)
        print("Threshold for type II errors:", threshold_alt)
        print("Ratio:", ratio)

        if threshold_alt >= threshold_null and ratio >= threshold_null:
            result = True
        else:
            result = False
        print("Result of the test:", result)
        return result
