import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

if __name__ == "__main__":
    # Exercise A: Fitting the half-normal model
    # Create a prior distribution for sigma such that:
    # - only positive values have nonzero probability
    # - the values of 1/sigma have a relatively wide range

    # prior for the model parameter sigma found through Mathematica: sigma ~ Exp(1/.3)
