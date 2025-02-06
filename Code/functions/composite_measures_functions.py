import numpy as np
import pandas as pd
from scipy import special
from scipy.stats import norm


def calculate_d_prime(df, signal, noise, n_signal, n_noise, col='correct_response', beta=False, c=False, Ad=False):
    """
    This function calculates d prime and uses the third correction for extreme values from this list:
    https://stats.stackexchange.com/questions/134779/d-prime-with-100-hit-rate-probability-and-0-false-alarm-probability
    To quote:
    add 0.5 to both the number of hits and the number of false alarms, and add 1 to both the number of signal trials and
    the number of noise trials; dubbed the loglinear approach (Hautus, 1995)
    Note: the loglinear method calls for adding 0.5 to all cells under the assumption that there are an equal number of
    signal and noise trials. If this is not the case, then the numbers will be different. If there are, say, 60% signal
    trials and 40% noise trials, then you would add 0.6 to the number of Hits, and 2x0.6 = 1.2 to the number of signal
    trials, and then 0.4 to the number of false alarms, and 2x0.4 = 0.8 to the number of noise trials, etc.

    The implementation of other measures follows https://lindeloev.net/calculating-d-in-python-and-php/

    The function requires (beside numpy and pandas):
    from scipy import special
    from scipy.stats import norm

    Parameters
    ----------
    df: dataframe, assumes columns with response_type labels ['hit', 'CR', 'FA', 'miss'] and a column with correct
        response per userID
    signal: float, adjusted ratio of n_signal/(2*n_trials_per_form_all) where n_signal is number of signal trials
    noise: float, adjusted ratio of n_noise/(2*n_trials_per_form_all) where n_noise is number of noise trials,
                 it must hold that: n_signal+n_noise == 2*n_trials_per_form_all
    n_signal: int, number of signal trials
    n_noise: int, number of noise trials

    Returns
    -------
    Series of d prime per userID calculated as df_d['hit'] - df_d['FA'] where these are ppfs of the adjusted rates

    """

    # select only relevant trials -- this is slow but I don't know how to make it faster
    df = df[df["response_type"].isin(['hit', 'FA'])]

    # get counts of hits and FAs (and all response types)
    # https://datascience.stackexchange.com/questions/94436/pandas-groupby-count-doesnt-count-zero-occurrences
    #     df_d = pd.crosstab(df["response_type"], df["userID"]).T
    # this is faster: https://stackoverflow.com/questions/37003100/pandas-groupby-for-zero-values/
    df_d = df.groupby(['response_type', 'userID'])[col].count().unstack(1, fill_value=0).T

    # make sure it works even if there is no hit or FA
    for col in ['hit', 'FA']:
        if col not in df_d.keys():
            df_d[col] = 0

    # the norm.ppf is slowing down the calculations, https://stackoverflow.com/questions/48552133/why-is-the-scipy-stats-norm-ppf-implementation-so-slow
    # df_d['FA'] = ((df_d['FA']+noise)/(n_noise+2*noise)).apply(lambda x: scipy.special.ndtri(x))
    # df_d['hit'] =((df_d['hit']+signal)/(n_signal+2*noise)).apply(lambda x: scipy.special.ndtri(x))

    # and list of comprehensions is also faster
    df_d['FA'] = [special.ndtri(x) for x in (df_d['FA'] + noise) / (n_noise + 2 * noise)]
    df_d['hit'] = [special.ndtri(x) for x in (df_d['hit'] + signal) / (n_signal + 2 * signal)]

    # check if we asked for any of the measures, if yes, return dictionary, otherwise only return d prime to keep it fast
    if beta or c or Ad:
        # define dict
        out = {'d_prime': df_d['hit'] - df_d['FA']}
        # check one by one if we want those measures
        if beta:
            out['beta'] = np.exp((df_d['FA']**2 - df_d['hit']**2)/2) #math.exp((Z(fa_rate) ** 2 - Z(hit_rate) ** 2) / 2),
        if c:
            out['c'] = -(df_d['hit'] + df_d['FA']) / 2  # -(Z(hit_rate) + Z(fa_rate)) / 2,
        if Ad:
            out['Ad'] = (out['d_prime']/np.sqrt(2)).apply(lambda x: norm.cdf(x)) #norm.cdf(out['d'] / math.sqrt(2)),

        return out

    return df_d['hit'] - df_d['FA']


def get_z_value(p_value, tailed='two'):
    """
    Calculate the z-value corresponding to a given p-value for a one-tailed or two-tailed test.

    Parameters
    ----------
    p_value: float, the p-value that I want to get (e.g., 0.05)
    tailed: str, optional, default is two, 'one' for one-tailed test, 'two' for two-tailed test

    Returns
    ----------
    corresponding z-value
    """
    if tailed == 'two':
        # for two-tailed test, divide p-value by 2 for each tail
        z_value = norm.ppf(1 - p_value / 2)
    elif tailed == 'one':
        z_value = norm.ppf(1 - p_value)
    else:
        raise ValueError("Invalid value for 'tailed'. Use 'one' or 'two'.")

    return z_value