"""
CS-134/PSY-141: Cognitive Modeling
Homework 1: Modeling The Bystander Effect
Crista Falk 10/3/2025
"""

# imports
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

# Exercise A – Modeling the Bystander Effect

def bystander_prediction(A, N):
    """
    Given N bystanders and a population "altruism" of A, return the probability of rescue across a list of data points.
    This model uses an exponential distribution function over the number of bystanders, scaled by A.

    Inputs: N: np.array[(int)] <- number of bystanders in a list of data points
            A: np.array[(float in range 0:1)] <- corresponding Altruism levels
    Outputs: P: np.array[(float in range 0:1)] <- corresponding predicted probability that this person is rescued
    """

    # predicted survival probability exponentially decreases when N increases and increases when A increases
    prediction = A/8 * math.e ** (-A/8*N)

    # if there are no bystanders, the person will definitely not be saved, so we replace all N=0 items with 0 in prediction
    np.where(N == 0, 0, prediction)

    # otherwise the model returns P(survival)
    return prediction

# Exercise D – The likelihood

def minus_log_likelihood(A, filename="empirical.csv"):
    """
    Given a filename return the negative log-likelihood of the model.
    Inputs: filename (str) <- the name of the file
    Outputs: nll (float) <- negative log-likelihood
    """

    df = pd.read_csv(filename)
    actual_bystanders = df.NUMBER_OF_BYSTANDERS
    actual_survival = df.RESCUED

    predicted_survival = bystander_prediction(A, actual_bystanders)

    # probability density function of the model's output distribution for the actual result
    nll = np.sum(-actual_survival * np.log(predicted_survival) - (1 - actual_survival) * np.log(1 - predicted_survival))

    return nll

# Main function will walk through entire assignment
if __name__ == '__main__':
    ### Exercise A

    # parameter test range
    A_range = np.linspace(0, 1, 11)
    N_range = np.linspace(0, 50, 11)

    # show predictions in text output
    for A in A_range:
        for N in N_range:
            print(f"P(Survival| A={A}, N={N}) = ", bystander_prediction(A, N))
        print("--")

    # visualize how the probability of survival changes as N or A changes
    A_values, N_values = np.meshgrid(A_range, N_range)
    P_values = bystander_prediction(A_values, N_values)
    # we want to vectorize our model to get many predictions at once given many possible input combinations
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(A_values, N_values, P_values, cmap='RdBu')
    ax.set_xlabel('Altruism Level')
    ax.set_ylabel('Number of Bystanders')
    ax.set_zlabel('Probability of Survival')
    plt.show()

    ### Exercise D
    print("NLL for supporting data:",minus_log_likelihood(0.5, "crista_falk_supporting_data.csv"))
    print("NLL for falsifying data:",minus_log_likelihood(0.5, "crista_falk_falsifying_data.csv"))
    print("NLL for empirical data:",minus_log_likelihood(0.5))

    ### Exercise E
    nll_values = [minus_log_likelihood(A=a) for a in A_range]
    print("NLL for empirical data:",nll_values)

    # plotting the results
    plt.plot(A_range, nll_values, color="purple")
    plt.title('Plotting -LL as function of A', fontsize=16, fontweight='bold')
    plt.xlabel('Altruism Level', fontsize=12)
    plt.ylabel('NLL of the Empirical Data', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()

    ### Bonus exercise
    # Use numerical optimization routines from existing Python libraries, e.g., the ones in the SciPy module
    res = minimize(minus_log_likelihood, 0.3, method='Nelder-Mead')
    print(res.x)

    # plotting optimal A
    plt.plot(A_range, nll_values, color="purple")
    plt.title('Plotting -LL as function of A', fontsize=16, fontweight='bold')
    plt.xlabel('Altruism Level', fontsize=12)
    plt.ylabel('NLL of the Empirical Data', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.axvline(x=res.x, color="red", linestyle="--", label=f"Optimal A={round(res.x[0],2)}")
    plt.legend()
    plt.tight_layout()
    plt.show()