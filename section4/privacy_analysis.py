from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import sys

from pate import *

def analyze_multiclass_confident_gnmax(
        votes, threshold, sigma_threshold, sigma_gnmax, budget, delta, file,
        show_dp_budget='disable', args=None):
    """
    Analyze how the pre-defined privacy budget will be exhausted when answering
    queries using the Confident GNMax mechanism.

    Args:
        votes: a 2-D numpy array of raw ensemble votes, with each row
        corresponding to a query.
        threshold: threshold value (a scalar) in the threshold mechanism.
        sigma_threshold: std of the Gaussian noise in the threshold mechanism.
        sigma_gnmax: std of the Gaussian noise in the GNMax mechanism.
        budget: pre-defined epsilon value for (eps, delta)-DP.
        delta: pre-defined delta value for (eps, delta)-DP.
        file: for logs.
        show_dp_budget: show the current cumulative dp budget.
        args: all args of the program

    Returns:
        max_num_query: when the pre-defined privacy budget is exhausted.
        dp_eps: a numpy array of length L = num-queries, with each entry
            corresponding to the privacy cost at a specific moment.
        partition: a numpy array of length L = num-queries, with each entry
            corresponding to the partition of privacy cost at a specific moment.
        answered: a numpy array of length L = num-queries, with each entry
            corresponding to the expected number of answered queries at a
            specific moment.
        order_opt: a numpy array of length L = num-queries, with each entry
            corresponding to the order minimizing the privacy cost at a
            specific moment.
    """
    max_num_query = 0

    def compute_partition(order_opt, eps):
        """Analyze how the current privacy cost is divided."""
        idx = np.searchsorted(orders, order_opt)
        rdp_eps_threshold = rdp_eps_threshold_curr[idx]
        rdp_eps_gnmax = rdp_eps_total_curr[idx] - rdp_eps_threshold
        p = np.array([rdp_eps_threshold, rdp_eps_gnmax,
                      -math.log(delta) / (order_opt - 1)])
        # assert sum(p) == eps
        # Normalize p so that sum(p) = 1
        return p / eps

    # RDP orders.
    orders = np.concatenate((np.arange(2, 100, .5),
                             np.logspace(np.log10(100), np.log10(1000),
                                         num=200)))
    # Number of queries
    n = len(votes)
    # All cumulative results
    dp_eps = np.zeros(n)
    partition = [None] * n
    order_opt = np.full(n, np.nan, dtype=float)
    answered = np.zeros(n, dtype=float)
    # Current cumulative results
    rdp_eps_threshold_curr = np.zeros(len(orders))
    rdp_eps_total_curr = np.zeros(len(orders))
    answered_curr = 0
    for i in range(n):
        v = votes[i]
        if sigma_threshold > 0:
            # logpr - probability that the label is answered.
            logpr = compute_logpr_answered(threshold, sigma_threshold, v)
            rdp_eps_threshold = compute_rdp_data_dependent_threshold(
                logpr, sigma_threshold, orders)
        else:
            # Do not use the Confident part of the GNMax.
            assert threshold == 0
            logpr = 0
            rdp_eps_threshold = 0

        logq = compute_logq_gnmax(v, sigma_gnmax)
        rdp_eps_gnmax = compute_rdp_data_dependent_gnmax(
            logq, sigma_gnmax, orders)
        rdp_eps_total = rdp_eps_threshold + np.exp(logpr) * rdp_eps_gnmax
        # Evaluate E[(rdp_eps_threshold + Bernoulli(pr) * rdp_eps_gnmax)^2]
        # Update current cumulative results.
        rdp_eps_threshold_curr += rdp_eps_threshold
        rdp_eps_total_curr += rdp_eps_total
        pr_answered = np.exp(logpr)
        answered_curr += pr_answered
        # Update all cumulative results.
        answered[i] = answered_curr
        dp_eps[i], order_opt[i] = rdp_to_dp(orders, rdp_eps_total_curr, delta)
        partition[i] = compute_partition(order_opt[i], dp_eps[i])
        # Verify if the pre-defined privacy budget is exhausted.
        if dp_eps[i] <= budget:
            max_num_query = i + 1
        else:
            break
        # Logs
        # if i % 100000 == 0 and i > 0:
        if show_dp_budget == 'apply':
            file = f'queries_answered_privacy_budget.txt'
            with open(file, 'a') as writer:
                if i == 0:
                    header = "queries answered,privacy budget"
                    print(header)
                    writer.write(f"{header}\n")
                info = f"{answered_curr},{dp_eps[i]}"
                print(info)
                writer.write(f"{info}\n")
                print(
                    'Number of queries: {} | E[answered]: {:.3f} | E[eps] at order {:.3f}: {:.4f} (contribution from delta: {:.4f})'.format(
                        i + 1, answered_curr, order_opt[i], dp_eps[i],
                        -math.log(delta) / (order_opt[i] - 1)))
                writer.write(
                    'Number of queries: {} | E[answered]: {:.3f} | E[eps] at order {:.3f}: {:.4f} (contribution from delta: {:.4f})\n'.format(
                        i + 1, answered_curr, order_opt[i], dp_eps[i],
                        -math.log(delta) / (order_opt[i] - 1)))
                sys.stdout.flush()
                writer.flush()

    return max_num_query, dp_eps, partition, answered, order_opt