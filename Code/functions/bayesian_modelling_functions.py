import os
from os.path import join as opj
import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt

def fit_model(formula, data, categorical_cols=['userID', 'task', 'difficulty'], base_lvl=None,
              check_priors=False, sample_priors=False, num_cpu=1, fit_kwargs=None, **kwargs):
    # deal with optional keyword arguments to the fitting
    if fit_kwargs is None:
        # it can't deal with None and the default should not be mutable, so if it's none, convert it here to {}
        fit_kwargs = {}
    elif not isinstance(fit_kwargs, dict):
        raise TypeError("fit_kwargs must be a dictionary.")

    # Model each trial individually with the same formula but adapting to the trial level
    model = bmb.Model(
        formula=formula,
        data=data,
        #         priors=priors,
        categorical=categorical_cols,
        dropna=True,
        **kwargs
    )

    if check_priors:
        # plot priors
        model.build()
        model.plot_priors()

    # fit model
    print(f'Using {num_cpu} core(s).')
    results = model.fit(idata_kwargs={"log_likelihood": True}, cores=num_cpu, **fit_kwargs)

    if sample_priors:
        # sample priors
        prior_samples = model.prior_predictive(draws=1000)
        az.plot_ppc(prior_samples, kind="kde", group="prior")
        plt.show()

    return model, results


def print_model(model, results, plot_priors=False):
    if plot_priors:
        # plot priors
        model.plot_priors()
        plt.show()

    # Plot posteriors
    az.plot_trace(
        results,
        #     var_names=["Intercept", "difficulty", "task", "1|userID", "task:difficulty", "sigma"],
        compact=True,
    )
    plt.show()

    # print summary
    #     az.summary(results, var_names=["Intercept", "difficulty", "task",  "task:difficulty", "sigma"])
    summary = az.summary(results)
    if len(summary > 20):
        print(summary.head(30), '\n...\n')
        print(summary.tail(30))
    else:
        print(summary)

    # show interaction if there is one
    if "task:difficulty" in list(results.posterior.data_vars):
        az.plot_forest(results, var_names=["task:difficulty"], combined=True, figsize=(8, 4))
        plt.axvline(0, c='k')

    elif "C(task, levels=base_lvl):difficulty" in list(results.posterior.data_vars):
        az.plot_forest(results, var_names=["C(task, levels=base_lvl):difficulty"], combined=True, figsize=(8, 4))
        plt.axvline(0, c='k')