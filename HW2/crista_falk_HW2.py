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
# exercises_to_run = ["A","B","C"]
exercises_to_run = ["A","B"]

# pymc global variables
RETURN_INFERENCEDATA = True
CORES = 8
CHAINS = 4
NSAMPLES = (
    10000  # Number of samples drawn using MCMC. Higher will make things more accurate!
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

    # -----------------------------
    # Exercise A: Fitting the half-normal model
    # -----------------------------

    # Create a prior distribution for sigma such that:
    # - only positive values have nonzero probability
    # - the values of 1/sigma have a relatively wide range

    # prior for the model parameter sigma found through Mathematica: sigma ~ Exp(1/mean_std)
    if "A" in exercises_to_run:
        mean_of_prior_stddev = 400
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

    # -----------------------------
    # Exercise B: Fitting a Gamma distribution
    # -----------------------------
    if "B" in exercises_to_run:
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
    if "A" and "B" and "C" in exercises_to_run:
        # compute a Bayes factor
        BF = az.compare({"GammaModel": trace_GammaModel, "HalfNormalModel": trace_HalfNormalModel})
        print("Bayes Factor:", BF)