__author__ = 'andrei'


import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, poisson


def rm_nans(np_array):
    """
    Flattens the np array and removes all the occurences of the np.NaN.
    destinated for functions working on flat arrays that don't tolerate NaN well
    :param np_array: input numpy array
    :return: flattened array without nans
    """
    fltr = np.logical_not(np.isnan(np_array))
    return np_array[fltr]


def Tukey_outliers(set_of_means, FDR=0.005, supporting_interval=0.5, verbose=False):
    """
    Performs Tukey quintile test for outliers from a normal distribution with defined false discovery rate
    :param set_of_means:
    :param FDR:
    :return:
    """
    # false discovery rate v.s. expected falses v.s. power
    q1_q3 = norm.interval(supporting_interval)
    FDR_q1_q3 = norm.interval(1 - FDR)  # TODO: this is not necessary: we can perfectly well fit it with proper params to FDR
    multiplier = (FDR_q1_q3[1] - q1_q3[1]) / (q1_q3[1] - q1_q3[0])
    l_means = len(set_of_means)

    q1 = np.percentile(set_of_means, 50*(1-supporting_interval))
    q3 = np.percentile(set_of_means, 50*(1+supporting_interval))
    high_fence = q3 + multiplier*(q3 - q1)
    low_fence = q1 - multiplier*(q3 - q1)

    if verbose:
        print 'FDR:', FDR
        print 'q1_q3', q1_q3
        print 'FDRq1_q3', FDR_q1_q3
        print 'q1, q3', q1, q3
        print 'fences', high_fence, low_fence

    if verbose:
        print "FDR: %s %%, expected outliers: %s, outlier 5%% confidence interval: %s"% (FDR*100, FDR*l_means,
                                                                                  poisson.interval(0.95, FDR*l_means))

    ho = (set_of_means < low_fence).nonzero()[0]
    lo = (set_of_means > high_fence).nonzero()[0]

    return lo, ho


def get_outliers(lane, FDR):
    """
    Gets the outliers in a lane with a given FDR and sets all non-outliers in the lane to NaNs
    :param lane:
    :param FDR:
    :return:
    """
    lo, ho = Tukey_outliers(lane, FDR)
    outliers = np.empty_like(lane)
    outliers.fill(np.nan)
    outliers[ho] = lane[ho]
    outliers[lo] = lane[lo]

    return outliers


def pull_breakpoints(contingency_list):
    """
    A method to extract breakpoints separating np.array regions with the same value.
    :param contingency_list: np.array containing regions of identical values
    :return: list of breakpoint indexes
    """
    no_nans_parsed = rm_nans(contingency_list)
    contingency = np.lib.pad(no_nans_parsed[:-1] == no_nans_parsed[1:], (1, 0), 'constant', constant_values=(True, True))
    nans_contingency = np.zeros(contingency_list.shape).astype(np.bool)
    nans_contingency[np.logical_not(np.isnan(contingency_list))] = contingency
    breakpoints = np.nonzero(np.logical_not(nans_contingency))[0].tolist()
    return breakpoints


def brp_setter(breakpoints_set, prebreakpoint_values):
    """
    Creates an array of the size defined by the biggest element of the breakpoints set and sets the
    intervals values to prebreakpoint_values. It assumes that the largest element of breakpoints set
    is equal to the size of desired array
    :param array:
    :param breakpoints_set:
    :param prebreakpoint_values:
    :return:
    """
    breakpoints_set = sorted(list(set(breakpoints_set))) # sorts the breakpoints
    assert(len(breakpoints_set) == len(prebreakpoint_values))
    support = np.empty((breakpoints_set[-1], )) # creates array to be filled
    support.fill(np.nan)

    pre_brp = 0 # fills the array
    for value, brp in zip(prebreakpoint_values, breakpoints_set):
        support[pre_brp:brp] = value
        pre_brp = brp

    return support


def local_min(a):
    return np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True]

def show_breakpoints(breakpoints, color = 'k'):
    for point in breakpoints:
        plt.axvline(x=point, color=color)