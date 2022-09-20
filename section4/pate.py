from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.stats
import math


def _logsumexp(x):
    """
    Sum in the log space.

    An addition operation in the standard linear-scale becomes the
    LSE (log-sum-exp) in log-scale.

    Args:
        x: array-like.

    Returns:
        A scalar.
    """
    x = np.array(x)
    m = max(x)  # for numerical stability
    return m + math.log(sum(np.exp(x - m)))


def _log1mexp(x):
    """
    Numerically stable computation of log(1-exp(x)).

    Args:
        x: a scalar.

    Returns:
        A scalar.
    """
    assert x <= 0, "Argument must be positive!"
    # assert x < 0, "Argument must be non-negative!"
    if x < -1:
        return math.log1p(-math.exp(x))
    elif x < 0:
        return math.log(-math.expm1(x))
    else:
        return -np.inf


def rdp_to_dp(orders, rdp, delta):
  """Compute epsilon given a list of RDP values and target delta.
  Args:
    orders: An array (or a scalar) of orders.
    rdp: A list (or a scalar) of RDP guarantees.
    delta: The target delta.
  Returns:
    Pair of (eps, optimal_order).
  Raises:
    ValueError: If input is malformed.
  """
  orders_vec = np.atleast_1d(orders)
  rdp_vec = np.atleast_1d(rdp)

  if delta <= 0:
    raise ValueError("Privacy failure probability bound delta must be >0.")
  if len(orders_vec) != len(rdp_vec):
    raise ValueError("Input lists must have the same length.")

  # Basic bound (see https://arxiv.org/abs/1702.07476 Proposition 3 in v3):
  #   eps = min( rdp_vec - math.log(delta) / (orders_vec - 1) )

  # Improved bound from https://arxiv.org/abs/2004.00010 Proposition 12 (in v4).
  # Also appears in https://arxiv.org/abs/2001.05990 Equation 20 (in v1).
  eps_vec = []
  for (a, r) in zip(orders_vec, rdp_vec):
    if a < 1:
      raise ValueError("Renyi divergence order must be >=1.")
    if r < 0:
      raise ValueError("Renyi divergence must be >=0.")

    if delta**2 + math.expm1(-r) >= 0:
      # In this case, we can simply bound via KL divergence:
      # delta <= sqrt(1-exp(-KL)).
      eps = 0  # No need to try further computation if we have eps = 0.
    elif a > 1.01:
      # This bound is not numerically stable as alpha->1.
      # Thus we have a min value of alpha.
      # The bound is also not useful for small alpha, so doesn't matter.
      eps = r + math.log1p(-1 / a) - math.log(delta * a) / (a - 1)
    else:
      # In this case we can't do anything. E.g., asking for delta = 0.
      eps = np.inf
    eps_vec.append(eps)

  idx_opt = np.argmin(eps_vec)
  return max(0, eps_vec[idx_opt]), orders_vec[idx_opt]

###############################
# RDP FOR THE GNMAX MECHANISM #
###############################
def compute_logq_gnmax(votes, sigma):
    """
    Computes an upper bound on log(Pr[outcome != argmax]) for the GNMax
    mechanism.

    Implementation of Proposition 7 from PATE 2018 paper.

    Args:
        votes: a 1-D numpy array of raw ensemble votes for a given query.
        sigma: std of the Gaussian noise in the GNMax mechanism.

    Returns:
        A scalar upper bound on log(Pr[outcome != argmax]) where log denotes natural logarithm.
    """
    num_classes = len(votes)
    variance = sigma ** 2
    idx_max = np.argmax(votes)
    votes_gap = votes[idx_max] - votes
    votes_gap = votes_gap[np.arange(num_classes) != idx_max]  # exclude argmax
    # Upper bound log(q) via a union bound rather than a more precise
    # calculation.
    logq = _logsumexp(
        scipy.stats.norm.logsf(votes_gap, scale=math.sqrt(2 * variance)))
    return min(logq,
               math.log(1 - (1 / num_classes)))  # another obvious upper bound


def compute_logq_multilabel_pate(labels_votes, sigma):
    """
    Computes an upper bound on log(Pr[outcome != S*]).

    Implementation of Proposition 7 from PATE 2018 paper + multilabel.

    Args:
        labels_votes: a 2-D numpy array of raw ensemble votes for a given query,
            with size (num_labels, num_classes), where num_classes=2 for the
            binary classification.
        sigma: std of the Gaussian noise in the GNMax mechanism.

    Returns:
        A scalar upper bound on log(Pr[outcome != S*]) where log denotes
        natural logarithm.
    """
    num_labels = len(labels_votes)
    all_logq = []
    for label in range(num_labels):
        votes = labels_votes[label]
        logq = compute_logq_gnmax(votes=votes, sigma=sigma)
        all_logq.append(logq)
    total_logq = _logsumexp(all_logq)
    # print(np.exp(total_logq)) # sum of probabilities
    # print(np.mean(np.exp(all_logq))) # mean of probability per label
    # return np.minimum(total_logq, 0)
    return total_logq


def compute_logq_gnmax_counting(votes, sigma, thresholds):
    """
    Computes an upper bound on log(Pr[outcome != S]) for the GNMax mechanism.

    Implementation of Proposition 1.3 / 1.4 from CaPC paper.

    Args:
        votes: a 1-D numpy array of raw ensemble votes for a given query.
        sigma: std of the Gaussian noise in the GNMax mechanism.

    Returns:
        A scalar upper bound on log(Pr[outcome != S]) where log denotes natural logarithm.
    """
    num_classes = len(votes)
    variance = sigma ** 2
    vector = votes >= thresholds
    gap = votes - thresholds
    logq = _logsumexp(
        np.concatenate(np.zeros(sum(vector)),
                       scipy.stats.norm.logsf(gap[vector == 1],
                                              scale=math.sqrt(variance)) ** (
                           -1),
                       scipy.stats.norm.logsf(gap[vector == 0],
                                              scale=math.sqrt(variance))))
    return min(logq,
               math.log(
                   1 - (1 / 2 ** (num_classes))))  # another obvious upper bound


def compute_rdp_data_independent_gnmax(sigma, orders):
    """
    Computes data-independent RDP guarantees for the GNMax mechanism.

    Args:
        sigma: std of the Gaussian noise in the GNMax mechanism.
        orders: an array-like list of RDP orders.

    Returns:
        A numpy array of upper bounds on RDP for all orders.

    Raises:
        ValueError: if the inputs are invalid.
    """
    if sigma < 0 or np.isscalar(orders) or np.any(orders <= 1):
        raise ValueError(
            "'sigma' must be non-negative, 'orders' must be array-like, "
            "and all elements in 'orders' must be greater than 1!")
    variance = sigma ** 2
    return np.array(orders) / variance


def compute_rdp_data_dependent_gnmax(logq, sigma, orders):
    """
    Computes data-dependent RDP guarantees for the GNMax mechanism.
    This is the bound D_\lambda(M(D) || M(D'))  from Theorem 6 (equation 2),
    PATE 2018 (Appendix A).

    Bounds RDP from above of GNMax given an upper bound on q.

    Args:
        logq: a union bound on log(Pr[outcome != argmax]) for the GNMax
            mechanism.
        sigma: std of the Gaussian noise in the GNMax mechanism.
        orders: an array-like list of RDP orders.

    Returns:
        A numpy array of upper bounds on RDP for all orders.

    Raises:
        ValueError: if the inputs are invalid.
    """
    if logq > 0 or sigma < 0 or np.isscalar(orders) or np.any(orders <= 1):
        raise ValueError(
            "'logq' must be non-positive, 'sigma' must be non-negative, "
            "'orders' must be array-like, and all elements in 'orders' must be "
            "greater than 1!")

    if np.isneginf(logq):  # deterministic mechanism with sigma == 0
        return np.full_like(orders, 0., dtype=np.float)

    variance = sigma ** 2
    orders = np.array(orders)
    rdp_eps = orders / variance  # data-independent bound as baseline

    # Two different higher orders computed according to Proposition 10.
    # See Appendix A in PATE 2018.
    # rdp_order2 = sigma * math.sqrt(-logq)
    rdp_order2 = math.sqrt(variance * -logq)
    rdp_order1 = rdp_order2 + 1

    # Filter out entries to which data-dependent bound does not apply.
    mask = np.logical_and(rdp_order1 > orders, rdp_order2 > 1)

    # Corresponding RDP guarantees for the two higher orders.
    # The GNMAx mechanism satisfies:
    # (order = \lambda, eps = \lambda / sigma^2)-RDP.
    rdp_eps1 = rdp_order1 / variance
    rdp_eps2 = rdp_order2 / variance

    log_a2 = (rdp_order2 - 1) * rdp_eps2

    # Make sure that logq lies in the increasing range and that A is positive.
    if (np.any(mask) and -logq > rdp_eps2 and logq <= log_a2 - rdp_order2 *
            (math.log(1 + 1 / (rdp_order1 - 1)) + math.log(
                1 + 1 / (rdp_order2 - 1)))):
        # Use log1p(x) = log(1 + x) to avoid catastrophic cancellations when x ~ 0.
        log1mq = _log1mexp(logq)  # log1mq = log(1-q)
        log_a = (orders - 1) * (
                log1mq - _log1mexp((logq + rdp_eps2) * (1 - 1 / rdp_order2)))
        log_b = (orders - 1) * (rdp_eps1 - logq / (rdp_order1 - 1))

        # Use logaddexp(x, y) = log(e^x + e^y) to avoid overflow for large x, y.
        log_s = np.logaddexp(log1mq + log_a, logq + log_b)

        # Values of q close to 1 could result in a looser bound, so minimum
        # between the data dependent bound and the data independent bound
        # rdp_esp = orders / variance is taken.
        rdp_eps[mask] = np.minimum(rdp_eps, log_s / (orders - 1))[mask]

    assert np.all(rdp_eps >= 0)
    return rdp_eps


def compute_rdp_data_dependent_gnmax_no_upper_bound(logq, sigma, orders):
    """
    If the data dependent bound applies, then use it even though its higher than
    the data independent bound. In this case, we are interested in estimating
    the privacy budget solely on the data and are not optimizing its value to be
    as small as possible.

    Computes data-dependent RDP guarantees for the GNMax mechanism.
    This is the bound D_\lambda(M(D) || M(D'))  from Theorem 6 (equation 2),
    PATE 2018 (Appendix A).

    Bounds RDP from above of GNMax given an upper bound on q.

    Args:
        logq: a union bound on log(Pr[outcome != argmax]) for the GNMax
            mechanism.
        sigma: std of the Gaussian noise in the GNMax mechanism.
        orders: an array-like list of RDP orders.

    Returns:
        A numpy array of upper bounds on RDP for all orders.

    Raises:
        ValueError: if the inputs are invalid.
    """
    if logq > 0 or sigma < 0 or np.isscalar(orders) or np.any(orders <= 1):
        raise ValueError(
            "'logq' must be non-positive, 'sigma' must be non-negative, "
            "'orders' must be array-like, and all elements in 'orders' must be "
            "greater than 1!")

    if np.isneginf(logq):  # deterministic mechanism with sigma == 0
        return np.full_like(orders, 0., dtype=np.float)

    variance = sigma ** 2
    orders = np.array(orders)
    rdp_eps = orders / variance  # data-independent bound as baseline

    # Two different higher orders computed according to Proposition 10.
    # See Appendix A in PATE 2018.
    # rdp_order2 = sigma * math.sqrt(-logq)
    rdp_order2 = math.sqrt(variance * -logq)
    rdp_order1 = rdp_order2 + 1

    # Filter out entries to which data-dependent bound does not apply.
    mask = np.logical_and(rdp_order1 > orders, rdp_order2 > 1)

    # Corresponding RDP guarantees for the two higher orders.
    # The GNMAx mechanism satisfies:
    # (order = \lambda, eps = \lambda / sigma^2)-RDP.
    rdp_eps1 = rdp_order1 / variance
    rdp_eps2 = rdp_order2 / variance

    log_a2 = (rdp_order2 - 1) * rdp_eps2

    # Make sure that logq lies in the increasing range and that A is positive.
    if (np.any(mask) and -logq > rdp_eps2 and logq <= log_a2 - rdp_order2 *
            (math.log(1 + 1 / (rdp_order1 - 1)) + math.log(
                1 + 1 / (rdp_order2 - 1)))):
        # Use log1p(x) = log(1 + x) to avoid catastrophic cancellations when x ~ 0.
        log1mq = _log1mexp(logq)  # log1mq = log(1-q)
        log_a = (orders - 1) * (
                log1mq - _log1mexp((logq + rdp_eps2) * (1 - 1 / rdp_order2)))
        log_b = (orders - 1) * (rdp_eps1 - logq / (rdp_order1 - 1))

        # Use logaddexp(x, y) = log(e^x + e^y) to avoid overflow for large x, y.
        log_s = np.logaddexp(log1mq + log_a, logq + log_b)

        # Do not apply the minimum between the data independent and data
        # dependent bound - but limit the computation to data dependent bound
        # only!
        rdp_eps[mask] = (log_s / (orders - 1))[mask]

    assert np.all(rdp_eps >= 0)
    return rdp_eps


def compute_rdp_data_independent_multilabel(sigma, orders, tau, norm):
    """
    Computes data-independent RDP guarantees for the Gaussian mechanism applied
    to the multilabel classification.

    Args:
        sigma: std of the Gaussian noise.
        orders: an array-like list of RDP orders.
        tau: tau clipping in a norm.
        norm: l2 or l1 norm

    Returns:
        A numpy array of upper bounds on RDP for all orders.

    Raises:
        ValueError: if the inputs are invalid.
    """
    if sigma < 0 or np.isscalar(orders) or np.any(orders <= 1):
        raise ValueError(
            "'sigma' must be non-negative, "
            "'orders' must be array-like, and all elements in 'orders' must be "
            "greater than 1!")

    if sigma == 0:  # deterministic mechanism with sigma == 0
        return np.full_like(orders, 0., dtype=np.float)

    variance = sigma ** 2
    orders = np.array(orders)
    if norm == '1':
        sensitivity = (2 * tau) ** 2
    elif norm == '2':
        sensitivity = 2 * tau ** 2
    else:
        raise Exception(f"Unsupported norm: {norm}.")
    # data-independent bound
    rdp_eps = (orders * sensitivity) / (2 * variance)

    assert np.all(rdp_eps >= 0)
    return rdp_eps


def is_data_independent_rdp_always_opt_gnmax(num_teachers, num_classes, sigma,
                                             orders):
    """
    Tests whether data-independent bound is always optimal for the GNMax mechanism.

    Args:
        num_teachers: number of teachers.
        num_classes: number of classes.
        sigma: std of the Gaussian noise in the GNMax mechanism.
        orders: an array-like list of RDP orders.

    Returns:
        A boolean array of the same length as orders.
    """
    unanimous_votes = np.array([num_teachers] + [0] * (num_classes - 1))
    logq = compute_logq_gnmax(unanimous_votes, sigma)

    rdp_eps_dep = compute_rdp_data_dependent_gnmax(logq, sigma, orders)
    rdp_eps_ind = compute_rdp_data_independent_gnmax(sigma, orders)
    return np.isclose(rdp_eps_dep, rdp_eps_ind)


###################################
# RDP FOR THE THRESHOLD MECHANISM #
###################################


def compute_logpr_answered(threshold, sigma_threshold, votes):
    """
    Computes log(Pr[answered]) for the threshold mechanism.

    Args:
        threshold: the threshold (a scalar).
        sigma_threshold: std of the Gaussian noise in the threshold mechanism.
        votes: a 1-D numpy array of raw ensemble votes for a given query.

    Returns:
        The value of log(Pr[answered]) where log denotes natural logarithm.
    """
    return scipy.stats.norm.logsf(threshold - round(max(votes)),
                                  scale=sigma_threshold)


def compute_rdp_data_independent_threshold(sigma, orders):
    """
    Computes data-independent RDP gurantees for the threshold mechanism.

    Args:
        sigma: std of the Gaussian noise in the threshold mechanism.
        orders: an array-like list of RDP orders.

    Returns:
        A numpy array of upper bounds on RDP for all orders.

    Raises:
        ValueError: if the inputs are invalid.
    """
    # The input to the threshold mechanism has sensitivity 1 rather than 2 as
    # compared to the GNMax mechanism, hence the sqrt(2) factor below.
    return compute_rdp_data_independent_gnmax(2 ** .5 * sigma, orders)


def compute_rdp_data_dependent_threshold(logpr, sigma, orders):
    """
    Computes data-dependent RDP guarantees for the threshold mechanism.

    Args:
        logpr: the value of log(Pr[answered]) for the threshold mechanism.
        sigma: std of the Gaussian noise in the threshold mechanism.
        orders: an array-like list of RDP orders.

    Returns:
        A numpy array of upper bounds on RDP for all orders.

    Raises:
        ValueError: if the inputs are invalid.
    """
    logq = min(logpr, _log1mexp(logpr))
    # The input to the threshold mechanism has sensitivity 1 rather than 2 as
    # compared to the GNMax mechanism, hence the sqrt(2) factor below.
    return compute_rdp_data_dependent_gnmax(logq, 2 ** .5 * sigma, orders)


def is_data_independent_rdp_always_opt_threshold(num_teachers, t, sigma,
                                                 orders):
    """
    Tests whether data-independent bound is always optimal for the threshold mechanism.

    Args:
        num_teachers: number of teachers.
        t: the threshold (a scalar).
        sigma: std of the Gaussian noise in the threshold mechanism.
        orders: an array-like list of RDP orders.

    Returns:
        A boolean array of the same length as orders.
    """
    # Since the data-dependent bound depends only on max(votes), it suffices to
    # check whether the data-dependent bounds are better than the data-independent
    # bounds in the extreme cases where max(votes) is minimal or maximal.
    # For both Confident GNMax and Interactive GNMax it holds that
    #   0 <= max(votes) <= num_teachers.
    logpr1 = compute_logpr_answered(t, sigma, np.array([0]))
    logpr2 = compute_logpr_answered(t, sigma, np.array([num_teachers]))

    rdp_eps_dep1 = compute_rdp_data_dependent_threshold(logpr1, sigma, orders)
    rdp_eps_dep2 = compute_rdp_data_dependent_threshold(logpr2, sigma, orders)

    rdp_eps_ind = compute_rdp_data_independent_threshold(sigma, orders)
    return np.logical_and(np.isclose(rdp_eps_dep1, rdp_eps_ind),
                          np.isclose(rdp_eps_dep2, rdp_eps_ind))


def create_orders():
    # RDP orders.
    orders = np.concatenate((np.arange(2, 100, .5),
                             np.logspace(np.log10(100), np.log10(1000),
                                         num=200)))
    return orders