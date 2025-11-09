import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import arviz as az
import multiprocessing
from pathlib import Path

def save_and_show(fig: mpl.figure.Figure, filename: str) -> None:
    """Save figure to plots/ then show and close."""
    fpath = PLOTS_DIR / filename
    fig.savefig(fpath, bbox_inches="tight", dpi=300)
    print(f"Saved: {fpath}")
    plt.show()
    plt.close(fig)

# remove or add elements to test only certain parts at a time
EXERCISES_TO_RUN = ["A","B","C"]
PRINT_GRAPHS = False

# pymc global variables
RETURN_INFERENCEDATA = True
CORES = 8
CHAINS = 4
NSAMPLES = (
    1000  # Number of samples drawn using MCMC. Higher will make things more accurate!
)

if __name__ == "__main__":
    # -----------------------------
    # Global config and helpers, courtesy of Muhammad Umair, 2025
    # -----------------------------
    GLOBAL_SEED = 42  # To be able to reproduce results.
    CONNECT_TO_DRIVE = False  # Set to true to save things to drive!

    # Ensure 'fork' (ignore if already set)
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    PLOTS_DIR = Path("plots")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # importing the data, convert to numpy from pandas
    df = pd.read_csv("exp2_part1.csv").RT
    data = df.to_numpy()

    # -----------------------------------------
    # Exercise A: Fitting the half-normal model
    # -----------------------------------------

    # Create a prior distribution for sigma such that:
    # - only positive values have nonzero probability
    # - the values of 1/sigma have a relatively wide range

    # prior for the model parameter sigma found through Mathematica: sigma ~ Exp(1/mean_std)
    if "A" in EXERCISES_TO_RUN:
        mean_of_prior_stddev = 500
        lambda_of_prior_stddev = 1/mean_of_prior_stddev

        with pm.Model() as HalfNormalModel:
            ## define our parameter priors, assuming that std is always positive
            std_RT = pm.Exponential("std_RT", lam=lambda_of_prior_stddev)

            # likelihood: this is "the model", defining probability of data given parameters.
            RT = pm.HalfNormal(
                "RT",  sigma=std_RT, observed=data
            )

            # Generating random data from the prior i.e., "prior predictive"
            prior_HalfNormalModel = pm.sample_prior_predictive(
                samples=4 * NSAMPLES,
                return_inferencedata=RETURN_INFERENCEDATA
            )

            # updating step
            # simulator generates samples from the posterior.
            trace_HalfNormalModel = pm.sample(
                random_seed=GLOBAL_SEED,
                draws=NSAMPLES,
                cores=CORES,
                chains=CHAINS,
                return_inferencedata=RETURN_INFERENCEDATA,
            )
            # generate data from updated posterior, i.e., "posterior predictive"
            posterior_HalfNormalModel = pm.sample_posterior_predictive(
                trace_HalfNormalModel,
                return_inferencedata=RETURN_INFERENCEDATA
            )

            if PRINT_GRAPHS:
                # First Plot: RT Standard Deviation
                fig, ax = plt.subplots()
                az.plot_dist(prior_HalfNormalModel.prior.std_RT, color="red", ax=ax, label="Prior")
                az.plot_dist(
                    trace_HalfNormalModel.posterior.std_RT, color="blue", ax=ax, label="Posterior"
                )
                ax.set_title("Standard Deviation Model Parameter")
                ax.set_xlabel("RT (ms)")
                ax.set_ylabel("Density")
                ax.legend()
                save_and_show(fig, "RT_stddev_halfnormal.png")

                # Second Plot: RT Predictive
                fig, ax = plt.subplots()
                az.plot_dist(
                    prior_HalfNormalModel.prior_predictive.RT, color="red", ax=ax, label="Prior Predictive"
                )
                az.plot_dist(
                    posterior_HalfNormalModel.posterior_predictive.RT,
                    color="blue",
                    ax=ax,
                    label="Posterior Predictive",
                )
                az.plot_dist(data, color="green", ax=ax, label="Data")
                ax.set_xlabel("RT (ms)")
                ax.set_ylabel("Density")
                ax.set_title("Reaction Time Model")
                ax.set_xlim(0, 700)
                ax.set_xticks(np.linspace(0, 700, num=7))
                ax.legend()
                save_and_show(fig, "RT_predictive_halfnormal.png")

    # ----------------------------------------
    # Exercise B: Fitting a Gamma distribution
    # ----------------------------------------
    if "B" in EXERCISES_TO_RUN:
        # now the likelihood function will be modeled by a gamma distribution
        # 2 priors needed: one for mean and one for std dev

        with pm.Model() as GammaModel:
            # prior 1: mean RT - my guess is Normal centered strongly around ~350
            mu_RT = pm.Normal("mu_RT", mu=350, sigma=10)

            # prior 2: std dev of RT - uniform with wide range
            std_RT = pm.Uniform("std_RT", lower=50,upper=150)

            # likelihood: this is "the model", defining probability of data given parameters.
            RT = pm.Gamma(
                "RT", mu=mu_RT, sigma=std_RT, observed=data
            )

            # Generating random data from the prior i.e., "prior predictive"
            prior_GammaModel = pm.sample_prior_predictive(
                samples=4 * NSAMPLES,
                return_inferencedata=RETURN_INFERENCEDATA
            )

            # simulator generates samples from the posterior.
            trace_GammaModel = pm.sample(
                random_seed=GLOBAL_SEED,
                draws=NSAMPLES,
                cores=CORES,
                chains=CHAINS,
                return_inferencedata=RETURN_INFERENCEDATA,
            )
            # generate data from updated posterior, i.e., "posterior predictive"
            posterior_GammaModel = pm.sample_posterior_predictive(
                trace_GammaModel,
                return_inferencedata=RETURN_INFERENCEDATA
            )

        # generate plots for mean and std dev priors and posterior

        if PRINT_GRAPHS:

            fig, ax = plt.subplots()
            az.plot_dist(prior_GammaModel.prior.mu_RT, color="red", ax=ax, label="Prior")
            az.plot_dist(
                trace_GammaModel.posterior.mu_RT, color="blue", ax=ax, label="Posterior"
            )
            ax.set_title("Mean Model Parameter")
            ax.set_xlabel("RT (ms)")
            ax.set_ylabel("Density")
            ax.legend()
            save_and_show(fig, "RT_mean_gamma.png")

            fig, ax = plt.subplots()
            az.plot_dist(prior_GammaModel.prior.std_RT, color="red", ax=ax, label="Prior")
            az.plot_dist(
                trace_GammaModel.posterior.std_RT, color="blue", ax=ax, label="Posterior"
            )
            ax.set_title("Standard Deviation Model Parameter")
            ax.set_xlabel("RT (ms)")
            ax.set_ylabel("Density")
            ax.legend()
            save_and_show(fig, "RT_stddev_gamma.png")

            # generate prior predictive, posterior predictive, and data historgram
            fig, ax = plt.subplots()
            az.plot_dist(
                prior_GammaModel.prior_predictive.RT, color="red", ax=ax, label="Prior Predictive"
            )
            az.plot_dist(
                posterior_GammaModel.posterior_predictive.RT,
                color="blue",
                ax=ax,
                label="Posterior Predictive",
            )
            az.plot_dist(data, color="green", ax=ax, label="Data")
            ax.set_xlabel("RT (ms)")
            ax.set_ylabel("Density")
            ax.set_title("Reaction Time Model")
            ax.set_xlim(0, 700)
            ax.set_xticks(np.linspace(0, 700, num=7))
            ax.legend()
            save_and_show(fig, "RT_predictive_gamma.png")

    # -----------------------------
    # Exercise C: Model Comparison
    # -----------------------------
    if "A" and "B" and "C" in EXERCISES_TO_RUN:
        # compute a Bayes factor

        # estimates the marginal likelihood for a model
        def marginal_llk_smc(
                model: pm.Model, n_samples: int, cores: int, chains: int, seed: int
        ) -> float:
            """
            Compute the marginal log likelihood using the sequential monte carlo (SMC)
            sampler.

            Parameters
            ----------
            model : pm.Model
            n_samples : int
            cores : int
            chains : int

            Returns
            -------
            float
                Marginal log likelihood estimate.
            """
            trace = pm.sample_smc(
                n_samples,
                model=model,
                cores=cores,
                chains=chains,
                return_inferencedata=True,
                random_seed=seed,
            )

            try:
                return trace.sample_stats["log_marginal_likelihood"].mean().item()
            except:
                raise ValueError("Unable to compute BF due to convergence error")


        def bayes_factor(
                model1: pm.Model,
                model2: pm.Model,
                n_samples: int = 15000,
                cores: int = 1,
                chains: int = 4,
                seed: int = None,
        ) -> float:
            """
            Compute an estimate of the Bayes factor p(y|model1) / p(y|model2).
            NOTE: Only pymc > 3 is supported.

            Parameters
            ----------
            model1 : pm.Model
                The model in the numerator of the bayes factor.
            model2 : pm.Model
                The model in the denominator of the bayes factor
            n_samples : int, optional
                Number of samples to draw during estimate, by default 5000
            cores : int, optional
                Number of cores to use when generating trace, by default 1
            Returns
            -------
            float
                The bayes factor estimate
            """

            # Compute the log marginal likelihoods for the models
            log_marginal_ll1 = marginal_llk_smc(
                model=model1, n_samples=n_samples, cores=cores, chains=chains, seed=seed
            )
            log_marginal_ll2 = marginal_llk_smc(
                model=model2, n_samples=n_samples, cores=cores, chains=chains, seed=seed
            )
            return np.exp(log_marginal_ll1 - log_marginal_ll2)


        BF = bayes_factor(
            model1=GammaModel,
            model2=HalfNormalModel,
            n_samples=10000,  # These are the number of samples we want to use for the computation. Higher the better.
            cores=CORES,
            chains=4,
            seed=GLOBAL_SEED,
        )

        def report_bf(bf, model1_name: str, model2_name: str) -> None:
            """
            Parameters
            -----------
            bf : float
              Bayes factor value
            model1_name : str
              Name of the model in the numerator of the BF
            model2_name : str
              Name of the model in the denominator of the BF
            """

            model_names = [model1_name, model2_name]
            bf_curr = bf
            if bf < 1:
                bf_curr = 1 / bf
                model_names.reverse()
            print(
                f"The data is more likely under the '{model_names[0]}' than the '{model_names[1]}', BF_{model1_name}_{model2_name}={bf}\n"
            )

            print(
                f"BF_{model2_name}_{model1_name}={1/bf}\n"
            )

        report_bf(BF, "gamma", "halfNormal")