import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Step 1: Load JSON Data
with open("../data/isoflops_curves.json", "r") as file:
    data = json.load(file)

# Step 2: Organize Data by Compute Budget
compute_budgets = {}
for run in data:
    budget = run["compute_budget"]
    if budget not in compute_budgets:
        compute_budgets[budget] = []
    compute_budgets[budget].append(run)

# Step 3: Find the Lowest Loss for Each Compute Budget
lowest_loss_runs = []
for budget, runs in compute_budgets.items():
    best_run = min(runs, key=lambda x: x["final_loss"])
    lowest_loss_runs.append(best_run)

# Extract parameters and final losses for fitting
parameters = np.array([run["parameters"] for run in lowest_loss_runs])
losses = np.array([run["final_loss"] for run in lowest_loss_runs])

# Step 4: Define a Quadratic Function for Curve Fitting
def quadratic_function(x, a, b):
    return a * x**b


# Step 5: Fit the Curve
popt, pcov = curve_fit(quadratic_function, parameters, losses)
a, b = popt

# Step 6: Visualize the Results
plt.scatter(parameters, losses, color="blue", label="Data Points")
x_fit = np.linspace(min(parameters), max(parameters), 500)
y_fit = quadratic_function(x_fit, *popt)
plt.plot(x_fit, y_fit, color="red", label=f"Fit: Loss = {a:.3e} * x^{b:.3e}")
plt.xlabel("Parameters")
plt.ylabel("Final Loss")
plt.title("IsoFLOPs Method - Scaling Law Fit")
plt.legend()
plt.savefig("../figures/isoflops_scaling.png")
plt.show()


# Step 7: Print Fit Results
print(f"Fitted quadratic coefficients: a={a:.3e}, b={b:.3e}")