# All of this is from the original code at: https://github.com/jparkerholder/procgen_autorl
from __future__ import division, print_function

import logging
import math
from random import random

import GPy

# import matplotlib.pyplot as plt
import numpy as np

# import scipy
from scipy import optimize as scipy_optimize
from sklearn import metrics as sklearn_metrics
from sklearn.metrics import pairwise as sklearn_metrics_pairwise

log = logging.getLogger(__name__)


class TV_SquaredExp(GPy.kern.Kern):
    """Time varying squared exponential kernel.
    For more info see the TV-GP-UCB paper:
    http://proceedings.mlr.press/v51/bogunovic16.pdf
    """

    def __init__(self, input_dim, variance=1.0, lengthscale=1.0, epsilon=0.0, active_dims=None):
        super().__init__(input_dim, active_dims, "time_se")
        self.variance = GPy.core.Param("variance", variance)
        self.lengthscale = GPy.core.Param("lengthscale", lengthscale)
        self.epsilon = GPy.core.Param("epsilon", epsilon)
        self.link_parameters(self.variance, self.lengthscale, self.epsilon)

    def K(self, X, X2):
        # time must be in the far left column
        if self.epsilon > 0.5:  # 0.5
            self.epsilon = 0.5
        if X2 is None:
            X2 = np.copy(X)
        T1 = X[:, 0].reshape(-1, 1)
        T2 = X2[:, 0].reshape(-1, 1)
        dists = sklearn_metrics.pairwise_distances(T1, T2, "cityblock")
        timekernel = (1 - self.epsilon) ** (0.5 * dists)

        X = X[:, 1:]
        X2 = X2[:, 1:]

        RBF = self.variance * np.exp(-np.square(sklearn_metrics_pairwise.euclidean_distances(X, X2)) / self.lengthscale)

        return RBF * timekernel

    def Kdiag(self, X):
        return self.variance * np.ones(X.shape[0])

    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None:
            X2 = np.copy(X)
        T1 = X[:, 0].reshape(-1, 1)
        T2 = X2[:, 0].reshape(-1, 1)

        X = X[:, 1:]
        X2 = X2[:, 1:]
        dist2 = np.square(sklearn_metrics_pairwise.euclidean_distances(X, X2)) / self.lengthscale

        dvar = np.exp(-np.square((sklearn_metrics_pairwise.euclidean_distances(X, X2)) / self.lengthscale))
        dl = -(
            2 * sklearn_metrics_pairwise.euclidean_distances(X, X2) ** 2 * self.variance * np.exp(-dist2)
        ) * self.lengthscale ** (-2)
        n = sklearn_metrics.pairwise_distances(T1, T2, "cityblock") / 2
        deps = -n * (1 - self.epsilon) ** (n - 1)

        self.variance.gradient = np.sum(dvar * dL_dK)
        self.lengthscale.gradient = np.sum(dl * dL_dK)
        self.epsilon.gradient = np.sum(deps * dL_dK)


class TV_MixtureViaSumAndProduct(GPy.kern.Kern):
    """Time varying mixture kernel from CoCaBO:
    http://proceedings.mlr.press/v119/ru20a.html
    """

    def __init__(
        self,
        input_dim,
        variance_1=1.0,
        variance_2=1.0,
        variance_mix=1.0,
        lengthscale=1.0,
        epsilon_1=0.0,
        epsilon_2=0.0,
        mix=0.5,
        cat_dims=[],
        active_dims=None,
    ):
        super().__init__(input_dim, active_dims, "time_se")

        self.cat_dims = cat_dims

        self.variance_1 = GPy.core.Param("variance_1", variance_1)
        self.variance_2 = GPy.core.Param("variance_2", variance_2)
        self.lengthscale = GPy.core.Param("lengthscale", lengthscale)
        self.epsilon_1 = GPy.core.Param("epsilon_1", epsilon_1)
        self.epsilon_2 = GPy.core.Param("epsilon_2", epsilon_2)
        self.mix = GPy.core.Param("mix", mix)
        # self.variance_mix = Param("variance_mix", variance_mix)
        self.variance_mix = variance_mix  # fixed

        self.link_parameters(
            self.variance_1,
            self.variance_2,
            self.lengthscale,
            self.epsilon_1,
            self.epsilon_2,
            # self.variance_mix,
            self.mix,
        )

    def prepare_data(self, X, X2):
        T1 = X[:, 0].reshape(-1, 1)
        T2 = X2[:, 0].reshape(-1, 1)

        X = X[:, 1:]
        X2 = X2[:, 1:]

        # shift becase we have removed time
        cat_dims = [x - 1 for x in self.cat_dims]

        X_cat = X[:, cat_dims]
        X_cont = X[:, [x for x in range(X.shape[1]) if x not in cat_dims]]

        X2_cat = X2[:, cat_dims]
        X2_cont = X2[:, [x for x in range(X2.shape[1]) if x not in cat_dims]]

        return T1, T2, X_cat, X_cont, X2_cat, X2_cont

    def K1(self, X, X2):
        # format data
        if X2 is None:
            X2 = np.copy(X)

        T1, T2, X_cat, X_cont, X2_cat, X2_cont = self.prepare_data(X, X2)

        # time kernel k_t
        dists = sklearn_metrics.pairwise_distances(T1, T2, "cityblock")
        timekernel_1 = (1 - self.epsilon_1) ** (0.5 * dists)

        # SE kernel k_se
        RBF = self.variance_1 * np.exp(
            -np.square(sklearn_metrics_pairwise.euclidean_distances(X_cont, X2_cont)) / self.lengthscale
        )

        # k1 = k_se * k_t
        k1 = RBF * timekernel_1

        return k1

    def K2(self, X, X2):
        # format data
        if X2 is None:
            X2 = np.copy(X)

        T1, T2, X_cat, X_cont, X2_cat, X2_cont = self.prepare_data(X, X2)

        # time kernel k_t
        dists = sklearn_metrics.pairwise_distances(T1, T2, "cityblock")
        timekernel_2 = (1 - self.epsilon_2) ** (0.5 * dists)

        # CategoryOverlapKernel
        # convert cat to int so we can subtract
        cat_vals = list(set(X_cat.flatten()).union(set(X2_cat.flatten())))
        for i, val in enumerate(cat_vals):
            X_cat = np.where(X_cat == val, i, X_cat)
            X2_cat = np.where(X2_cat == val, i, X2_cat)
        diff = X_cat[:, None] - X2_cat[None, :]
        diff[np.where(np.abs(diff))] = 1
        diff1 = np.logical_not(diff)
        k_cat = self.variance_2 * np.sum(diff1, -1) / len(self.cat_dims)

        # k2 = k_cat * k_t
        k2 = k_cat * timekernel_2

        return k2

    def K(self, X, X2):
        # clip epsilons
        if self.epsilon_1 > 0.5:  # 0.5
            self.epsilon_1 = 0.5

        if self.epsilon_2 > 0.5:  # 0.5
            self.epsilon_2 = 0.5

        # format data
        if X2 is None:
            X2 = np.copy(X)

        k1 = self.K1(X, X2)
        k2 = self.K2(X, X2)

        # K_mix
        k_out = self.variance_mix * ((1 - self.mix) * 0.5 * (k1 + k2) + self.mix * k1 * k2)

        return k_out

    def Kdiag(self, X):
        """
        Not sure what this is for?
        """
        return np.ones(X.shape[0])

    def update_gradients_full(self, dL_dK, X, X2):
        # format data

        if X2 is None:
            X2 = np.copy(X)

        k1_xx = self.K1(X, X2)
        k2_xx = self.K2(X, X2)

        # K_mix = self.K(X, X2)

        T1, T2, X_cat, X_cont, X2_cat, X2_cont = self.prepare_data(X, X2)

        # compute common terms before K1 grads and K2 grads
        n = sklearn_metrics.pairwise_distances(T1, T2, "cityblock") / 2
        k_t1 = (1 - self.epsilon_1) ** (n - 1)
        k_t2 = (1 - self.epsilon_2) ** (n - 1)

        k_x = self.variance_1 * np.exp(
            -np.square(sklearn_metrics_pairwise.euclidean_distances(X_cont, X2_cont)) / self.lengthscale
        )

        # convert cat to int so we can subtract
        cat_vals = list(set(X_cat.flatten()).union(set(X2_cat.flatten())))
        for i, val in enumerate(cat_vals):
            X_cat = np.where(X_cat == val, i, X_cat)
            X2_cat = np.where(X2_cat == val, i, X2_cat)
        diff = X_cat[:, None] - X2_cat[None, :]
        diff[np.where(np.abs(diff))] = 1
        diff1 = np.logical_not(diff)
        k_h = self.variance_2 * np.sum(diff1, -1) / len(self.cat_dims)

        # K1 grads
        dist2 = np.square(sklearn_metrics_pairwise.euclidean_distances(X_cont, X2_cont)) / self.lengthscale

        dvar1 = np.exp(-np.square((sklearn_metrics_pairwise.euclidean_distances(X_cont, X2_cont)) / self.lengthscale))

        dl = -(
            sklearn_metrics_pairwise.euclidean_distances(X_cont, X2_cont) ** 2 * self.variance_1 * np.exp(-dist2)
        ) * self.lengthscale ** (-2)

        deps1 = -n * (1 - self.epsilon_1) ** (n - 1)
        dKout_l = (1 - self.mix) * k_t1 * dl + self.mix * self.K2(X, X2) * k_t1 * dl
        dKout_var1 = (1 - self.mix) * k_t1 * dvar1 + self.mix * self.K2(X, X2) * k_t1 * dvar1
        dKout_eps1 = (1 - self.mix) * k_x * deps1 + self.mix * self.K1(X, X2) * k_x * deps1

        self.variance_1.gradient = np.sum(dKout_var1 * dL_dK)
        self.lengthscale.gradient = np.sum(dKout_l * dL_dK)
        self.epsilon_1.gradient = np.sum(dKout_eps1 * dL_dK)

        # K2 grads
        dvar2 = np.sum(diff1, -1) / len(self.cat_dims)

        deps2 = -n * (1 - self.epsilon_2) ** (n - 1)
        dKout_var2 = (1 - self.mix) * k_t2 * dvar2 + self.mix * self.K1(X, X2) * k_t2 * dvar2
        dKout_eps2 = (1 - self.mix) * k_h * deps2 + self.mix * self.K2(X, X2) * k_h * deps2

        self.variance_2.gradient = np.sum(dKout_var2 * dL_dK)
        self.epsilon_2.gradient = np.sum(dKout_eps2 * dL_dK)

        # K_mix grads

        self.mix.gradient = np.sum(dL_dK * (-(k1_xx + k2_xx) + (k1_xx * k2_xx)))

        # self.variance_mix.gradient = \
        #    np.sum(K_mix * dL_dK) / self.variance_mix


def normalize(data, wrt):
    """Normalize data to be in range (0,1), with respect to (wrt) boundaries,
    which can be specified.
    """
    return (data - np.min(wrt, axis=0)) / (np.max(wrt, axis=0) - np.min(wrt, axis=0))


def standardize(data):
    """Standardize to be Gaussian N(0,1). Clip final values."""
    data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
    return np.clip(data, -2, 2)


def UCB(m, m1, x, fixed, kappa=0.5):
    """UCB acquisition function. Interesting points to note:
    1) We concat with the fixed points, because we are not optimizing wrt
       these. This is the Reward and Time, which we can't change. We want
       to find the best hyperparameters *given* the reward and time.
    2) We use m to get the mean and m1 to get the variance. If we already
       have trials running, then m1 contains this information. This reduces
       the variance at points currently running, even if we don't have
       their label.
       Ref: https://jmlr.org/papers/volume15/desautels14a/desautels14a.pdf
    """

    c1 = 0.2
    c2 = 0.4
    beta_t = np.max([c1 * np.log(c2 * m.X.shape[0]), 0])
    kappa = np.sqrt(beta_t)

    xtest = np.concatenate((fixed.reshape(-1, 1), np.array(x).reshape(-1, 1))).T

    try:
        preds = m.predict(xtest)
        preds = m.predict(xtest)
        mean = preds[0][0][0]
    except ValueError:
        mean = -9999

    try:
        preds = m1.predict(xtest)
        var = preds[1][0][0]
    except ValueError:
        var = 0
    return mean + kappa * var


def optimize_acq(func, m, m1, fixed, num_f):
    """Optimize acquisition function."""

    opts = {"maxiter": 200, "maxfun": 200, "disp": False}

    T = 10
    best_value = -999
    best_theta = m1.X[0, :]

    bounds = [(0, 1) for _ in range(m.X.shape[1] - num_f)]

    for ii in range(T):
        x0 = np.random.uniform(0, 1, m.X.shape[1] - num_f)

        res = scipy_optimize.minimize(
            lambda x: -func(m, m1, x, fixed),
            x0,
            bounds=bounds,
            method="L-BFGS-B",
            options=opts,
        )

        val = func(m, m1, res.x, fixed)
        if val > best_value:
            best_value = val
            best_theta = res.x

    return np.clip(best_theta, 0, 1)


def select_length(Xraw, yraw, bounds, num_f):
    """Select the number of datapoints to keep, using cross validation"""
    min_len = 200

    if Xraw.shape[0] < min_len:
        return Xraw.shape[0]
    else:
        length = min_len - 10
        scores = []
        while length + 10 <= Xraw.shape[0]:
            length += 10

            base_vals = np.array(list(bounds.values())).T
            X_len = Xraw[-length:, :]
            y_len = yraw[-length:]
            oldpoints = X_len[:, :num_f]
            old_lims = np.concatenate((np.max(oldpoints, axis=0), np.min(oldpoints, axis=0))).reshape(
                2, oldpoints.shape[1]
            )
            limits = np.concatenate((old_lims, base_vals), axis=1)

            X = normalize(X_len, limits)
            y = standardize(y_len).reshape(y_len.size, 1)

            kernel = TV_SquaredExp(input_dim=X.shape[1], variance=1.0, lengthscale=1.0, epsilon=0.1)
            m = GPy.models.GPRegression(X, y, kernel)
            m.optimize(messages=True)

            scores.append(m.log_likelihood())
        idx = np.argmax(scores)
        length = (idx + int((min_len / 10))) * 10
        return length


def estimate_alpha(batch_size, gamma, Wc, C):
    def single_evaluation(alpha):
        denominator = sum([alpha if val > alpha else val for idx, val in enumerate(Wc)])
        rightside = (1 / batch_size - gamma / C) / (1 - gamma)
        output = np.abs(alpha / denominator - rightside)

        return output

    x_tries = np.random.uniform(0, np.max(Wc), size=(100, 1))
    y_tries = [single_evaluation(val) for val in x_tries]
    # find x optimal for init
    # print(f'ytry_len={len(y_tries)}')
    idx_min = np.argmin(y_tries)
    x_init_min = x_tries[idx_min]

    res = scipy_optimize.minimize(
        single_evaluation,
        x_init_min,
        method="BFGS",
        options={"gtol": 1e-6, "disp": False},
    )
    if isinstance(res, float):
        return res
    else:
        return res.x


def timevarying_compute_prob_dist_and_draw_hts(weights, gamma, batch_size, omega, pending_actions):
    # number of category
    C = len(weights)

    if batch_size <= 1:
        print("batch_size needs to be >1")

    # perform some truncation here
    maxW = np.max(weights)
    eta = (1 / batch_size - gamma / C) / (1 - gamma)
    temp = np.sum(weights) * eta  # (1.0 / batch_size - gamma / C) / (1 - gamma)
    if gamma < 1 and maxW >= temp and batch_size < C:
        # find a threshold alpha
        alpha = estimate_alpha(batch_size, gamma, weights, C)

        S0 = [idx for idx, val in enumerate(weights) if val > alpha]
        S1 = [idx for idx in pending_actions if (weights[idx] > alpha)]
        S0 += S1
        # update Wc_list
        for idx in S0:
            weights[idx] = alpha[0]
    else:
        S0 = []

    e_num = 2.71  # this is e number
    # Compute the probability for each category
    probabilityDistribution = distrEXP3M(weights, gamma) + e_num * omega * np.sum(weights) / C
    # print("prob",np.round(probabilityDistribution,decimals=4))

    # draw a batch here
    if batch_size < C:
        # we need to multiply the prob by batch_size before providing into DepRound
        probabilityDistribution = [prob * batch_size for prob in probabilityDistribution]
        myselection = DepRound(probabilityDistribution, k=batch_size)

    else:
        probabilityDistribution = np.nan_to_num(probabilityDistribution)
        if sum(probabilityDistribution) == 0:
            probabilityDistribution += 1
        probabilityDistribution = probabilityDistribution / np.sum(probabilityDistribution)
        myselection = np.random.choice(len(probabilityDistribution), batch_size, p=probabilityDistribution)
        myselection = myselection.tolist()

    return myselection, probabilityDistribution, S0


def distrEXP3M(weights, gamma=0.0):
    # given the weight vector and gamma, return the distribution
    theSum = float(sum(weights))
    return [(1.0 - gamma) * (w / theSum) + (gamma / len(weights)) for w in weights]


def exp3_get_cat(row, data, numRounds, index, pop_size):
    arms = row
    numActions = len(arms)
    batch_size = len(data.T[0])

    # pendingactions = [arms.index(x) for x in pendingactions]

    if batch_size < numActions:
        gamma = math.sqrt(numActions * np.log(numActions / batch_size) / ((np.e - 1) * batch_size * (numRounds / 10)))
    else:
        gamma = 0.2

    omega = 1 / (np.sum(numRounds) * 10)

    tt = 0

    weights = [1.0] * numActions
    # all_choice = []

    min_t = int(min(data.T[0]))
    max_t = int(max(data.T[0]))

    count = 0
    # choice = [0] * numRounds
    # all_choice = []

    # this is just where we build the distributions...
    for tt in range(min_t, max_t + 1):
        (batch_choice, probabilityDistribution, S0,) = timevarying_compute_prob_dist_and_draw_hts(
            weights, gamma, batch_size, omega, []  # pendingactions
        )

        batch_choice = []
        for i in range(batch_size):
            if data[i][0] == tt:
                batch_choice.append(arms.index(data[i][index]))
        # batch_choice = [
        #    arms.index(x) for x in data[tt]["x" + str(name)].values
        # ]

        batch_choice = np.asarray(batch_choice)

        e_num = 2.71  # e number
        right_term = e_num * omega * np.sum(weights) / numActions

        rewards = data.T[-1]

        for idx, val in enumerate(batch_choice):
            if val in S0:
                weights[val] += right_term
            else:
                # =============================================================================
                # this estimation of the reward comes from the RL...
                # the reward should be normalized [0-1] over time for the best performance....
                theReward = rewards[idx]

                estimatedReward = 1.0 * theReward / (probabilityDistribution[val] * batch_size)
                weights[val] *= (
                    np.exp(estimatedReward * gamma * batch_size / numActions) + right_term
                )  # important that we use estimated reward here!

        sum_w = np.sum(weights)
        weights = [w / sum_w for w in weights]

        count += 1

    # now we select our arm!

    (batch_choice, probabilityDistribution, S0,) = timevarying_compute_prob_dist_and_draw_hts(
        weights, gamma, batch_size, omega, []  # pendingactions
    )

    cat_idx = DepRound(probabilityDistribution, k=1)[0]
    cat = arms[cat_idx]
    log.info("Selecting categorical hyperparameters:")
    log.info(f"weights {np.round(weights, decimals=4)}")
    log.info(f"choices {arms}")
    log.info(f"dist {np.round(probabilityDistribution, decimals=4)}")
    log.info(f"next value {cat}")
    return cat


def with_proba(epsilon):
    """Bernoulli test, with probability `epsilon`, return `True`,
        and with probability `1 - epsilon`, return `False`.

    Example:
    >>> from random import seed; seed(0)  # reproductible
    >>> with_proba(0.5)
    False
    >>> with_proba(0.9)
    True
    >>> with_proba(0.1)
    False
    >>> if with_proba(0.2):
    ...     print("This happens 20% of the time.")
    """
    assert (
        0 <= epsilon <= 1
    ), "Error: for 'with_proba(epsilon)', epsilon = {:.3g} has to be between 0 and 1 to be a valid probability.".format(
        epsilon
    )  # DEBUG
    return random() < epsilon  # True with proba epsilon


# --- Utility functions
def DepRound(weights_p, k=1, isWeights=True):
    """
    [[Algorithms for adversarial bandit problems with multiple plays,
        by T.Uchiya, A.Nakamura and M.Kudo, 2010](http://hdl.handle.net/2115/47057)]
    Figure 5 (page 15) is a very clean presentation of the algorithm.
    - Inputs: `k < K` and weights_p `= (p_1, *, p_K)` such that :math:`sum_{i=1}^{K} p_i = k` (or `= 1`).
    - Output: A subset of :math:`{1,*,K}` with exactly `k` elements.
        Each action `i` is selected with probability exactly `p_i`.

    Example:
    >>> import numpy as np; import random
    >>> np.random.seed(0); random.seed(0)  # for reproductibility!
    >>> K = 5
    >>> k = 2
    >>> weights_p = [ 2, 2, 2, 2, 2 ]  # all equal weights
    >>> DepRound(weights_p, k)
    [3, 4]
    >>> DepRound(weights_p, k)
    [3, 4]
    >>> DepRound(weights_p, k)
    [0, 1]
    >>> weights_p = [ 10, 8, 6, 4, 2 ]  # decreasing weights
    >>> DepRound(weights_p, k)
    [0, 4]
    >>> DepRound(weights_p, k)
    [1, 2]
    >>> DepRound(weights_p, k)
    [3, 4]
    >>> weights_p = [ 3, 3, 0, 0, 3 ]  # decreasing weights
    >>> DepRound(weights_p, k)
    [0, 4]
    >>> DepRound(weights_p, k)
    [0, 4]
    >>> DepRound(weights_p, k)
    [0, 4]
    >>> DepRound(weights_p, k)
    [0, 1]
    - See [[Gandhi et al, 2006](http://dl.acm.org/citation.cfm?id=1147956)] for the details.
    """
    p = np.array(weights_p)
    K = len(p)
    # Checks
    assert k < K, "Error: k = {} should be < K = {}.".format(k, K)  # DEBUG
    if not np.isclose(np.sum(p), 1):
        p = p / np.sum(p)
    assert np.all(0 <= p) and np.all(
        p <= 1
    ), f"Error: the weights (p_1, ..., p_K) should all be 0 <= p_i <= 1 ..., is {p}"
    assert np.isclose(np.sum(p), 1), "Error: the sum of weights p_1 + ... + p_K should be = 1 (= {}).".format(
        np.sum(p)
    )  # DEBUG
    # Main loop
    possible_ij = [a for a in range(K) if 0 < p[a] < 1]
    while possible_ij:
        # Choose distinct i, j with 0 < p_i, p_j < 1
        if len(possible_ij) == 1:
            i = np.random.choice(possible_ij, size=1)
            j = i
        else:
            i, j = np.random.choice(possible_ij, size=2, replace=False)
        pi, pj = p[i], p[j]
        assert 0 < pi < 1, "Error: pi = {} (with i = {}) is not 0 < pi < 1.".format(pi, i)  # DEBUG
        assert 0 < pj < 1, "Error: pj = {} (with j = {}) is not 0 < pj < 1.".format(pj, i)  # DEBUG
        assert i != j, "Error: i = {} is different than with j = {}.".format(i, j)  # DEBUG

        # Set alpha, beta
        alpha, beta = min(1 - pi, pj), min(pi, 1 - pj)
        # proba = alpha / (alpha + beta)  #bug
        proba = beta / (alpha + beta)

        if with_proba(proba):  # with probability = proba = alpha/(alpha+beta)
            pi, pj = pi + alpha, pj - alpha
        else:  # with probability = 1 - proba = beta/(alpha+beta)
            pi, pj = pi - beta, pj + beta

        # Store
        p[i], p[j] = pi, pj
        # And update
        possible_ij = [a for a in range(K) if 0 < p[a] < 1]
        if len([a for a in range(K) if np.isclose(p[a], 0)]) == K - k:
            break
    # Final step
    subset = [a for a in range(K) if np.isclose(p[a], 1)]
    if len(subset) < k:
        subset = [a for a in range(K) if not np.isclose(p[a], 0)]
    assert (
        len(subset) == k
    ), "Error: DepRound({}, {}) is supposed to return a set of size {}, but {} has size {}...".format(
        weights_p, k, k, subset, len(subset)
    )  # DEBUG
    return subset


def DepRound2(weights_p, k=1):
    """
    [[Algorithms for adversarial bandit problems with multiple plays,
        by T.Uchiya, A.Nakamura and M.Kudo, 2010](http://hdl.handle.net/2115/47057)]
    Figure 5 (page 15) is a very clean presentation of the algorithm.
    - Inputs: :math:`k < K` and weights_p `= (p_1, *, p_K)` such that `sum_{i=1}^{K} p_i = k` (or :math:`= 1`).
    - Output: A subset of :math:`{1,*,K}` with exactly `k` elements.
        Each action `i` is selected with probability exactly `p_i`.
    Example:

    >>> import numpy as np; import random
    >>> np.random.seed(0); random.seed(0)  # for reproductibility!
    >>> K = 5
    >>> k = 2
    >>> weights_p = [ 2, 2, 2, 2, 2 ]  # all equal weights
    >>> DepRound(weights_p, k)
    [3, 4]
    >>> DepRound(weights_p, k)
    [3, 4]
    >>> DepRound(weights_p, k)
    [0, 1]
    >>> weights_p = [ 10, 8, 6, 4, 2 ]  # decreasing weights
    >>> DepRound(weights_p, k)
    [0, 4]
    >>> DepRound(weights_p, k)
    [1, 2]
    >>> DepRound(weights_p, k)
    [3, 4]
    >>> weights_p = [ 3, 3, 0, 0, 3 ]  # decreasing weights
    >>> DepRound(weights_p, k)
    [0, 4]
    >>> DepRound(weights_p, k)
    [0, 4]
    >>> DepRound(weights_p, k)
    [0, 4]
    >>> DepRound(weights_p, k)
    [0, 1]
    - See [[Gandhi et al, 2006](http://dl.acm.org/citation.cfm?id=1147956)] for the details.
    """
    p = np.array(weights_p)
    K = len(p)
    # Checks
    assert k < K, "Error: k = {} should be < K = {}.".format(k, K)  # DEBUG
    if not np.isclose(np.sum(p), 1):
        p = p / np.sum(p)
    assert np.all(0 <= p) and np.all(
        p <= 1
    ), f"Error: the weights (p_1, ..., p_K) should all be 0 <= p_i <= 1 ... but are {p}"
    assert np.isclose(np.sum(p), 1), "Error: the sum of weights p_1 + ... + p_K should be = 1 (= {}).".format(
        np.sum(p)
    )  # DEBUG
    # Main loop
    possible_ij = [a for a in range(K) if 0 < p[a] < 1]
    while possible_ij:
        # Choose distinct i, j with 0 < p_i, p_j < 1
        if len(possible_ij) == 1:
            i = np.random.choice(possible_ij, size=1)
            j = i
        else:
            i, j = np.random.choice(possible_ij, size=2, replace=False)
        pi, pj = p[i], p[j]
        assert 0 < pi < 1, "Error: pi = {} (with i = {}) is not 0 < pi < 1.".format(pi, i)  # DEBUG
        assert 0 < pj < 1, "Error: pj = {} (with j = {}) is not 0 < pj < 1.".format(pj, i)  # DEBUG
        assert i != j, "Error: i = {} is different than with j = {}.".format(i, j)  # DEBUG

        # Set alpha, beta
        alpha, beta = min(1 - pi, pj), min(pi, 1 - pj)
        # proba = alpha / (alpha + beta) #bug
        proba = beta / (alpha + beta)

        if with_proba(proba):  # with probability = proba = alpha/(alpha+beta)
            pi, pj = pi + alpha, pj - alpha
        else:  # with probability = 1 - proba = beta/(alpha+beta)
            pi, pj = pi - beta, pj + beta

        # Store
        p[i], p[j] = pi, pj
        # And update
        possible_ij = [a for a in range(K) if 0 < p[a] < 1]
        if len([a for a in range(K) if np.isclose(p[a], 0)]) == K - k:
            break
    # Final step
    subset = [a for a in range(K) if np.isclose(p[a], 1)]
    if len(subset) < k:
        subset = [a for a in range(K) if not np.isclose(p[a], 0)]
    assert (
        len(subset) == k
    ), "Error: DepRound({}, {}) is supposed to return a set of size {}, but {} has size {}...".format(
        weights_p, k, k, subset, len(subset)
    )  # DEBUG
    return subset
