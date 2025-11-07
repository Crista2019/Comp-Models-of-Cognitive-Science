# Comp-Models-of-Cognitive-Science

# installation steps
1. First download Anaconda (miniconda) from the website.
2. Run the following commands in the Anaconda terminal:
```
conda create --name pymc_518 python=3.10 -y
conda activate pymc_518
conda install m2w64-toolchain
pip install --no-cache-dir "pymc==5.18.2" 
pip install seaborn
```
3. Run the code itself, within this conda env, with the following command:
```python crista_falk_HW2.py```

# Exercise A: Fitting the half-normal model
Using an exponential prior over the mean standard deviation, I attempt to fit a Half Normal model to the data.

# Exercise B: Fitting a Gamma distribution
Using a Normal prior over the mean parameter and an uninformed, uniform prior over the standard deviation parameter, I attempt to fit a Gamma model to the data.

# Exercise C – Comparison and Discussion
Visually and numerically (using Bayes Factor), the Gamma distribution is better suited to the data.