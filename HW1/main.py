"""
CS-134/PSY-141: Cognitive Modeling
Homework 1: Modeling The Bystander Effect
Crista Falk 10/3/2025
"""

# imports
import math
import numpy as np
import matplotlib.pyplot as plt

# Exercise A – Modeling the Bystander Effect

def bystander_prediction(A, N):
    """
    Given N bystanders and a population "altruism" of A, return the probability of rescue.
    This model uses an exponential distribution function over the number of bystanders, scaled by A.

    Inputs: N (int) <- number of bystanders,
            A (float in range 0:1) <- Altruism level
    Outputs: P (float in range 0:1) <- Predicted probability that someone is rescued
    """

    if N.any() == 0:
        # if there are no bystanders, the person will definitely not be saved
        return 0
    # otherwise the model returns a predicted P(survival) which exponentially decreases when N increases and increases when A increases
    return A * math.e ** (-A*N)

# Exercise B – Supporting data

# Exercise C – Falsifying data

# Exercise D – the likelihood

# Exercise E – Plotting MLL as function of A

# Main function will walk through entire assignment
if __name__ == '__main__':
    # Exercise A

    # parameter test range
    A_range = np.linspace(0, 1, 5)
    N_range = np.linspace(0, 4, 5)

    # show predictions in text
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

    # Exercise B

    # Exercise C

    # Exercise D

    # Exercise E
    pass

