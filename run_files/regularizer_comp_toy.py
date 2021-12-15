import numpy as np
import stew.example_data as create
import stew.mlogit as mlogit
import stew.utils as utils
import matplotlib.pyplot as plt
from stew.utils import create_diff_matrix
from sklearn.linear_model import LinearRegression
from stew.regression import *
import matplotlib.pyplot as plt
import os
from datetime import datetime
# from stew.regression import LinearRegressionTorch

time_id = datetime.now().strftime('%Y_%m_%d_%H_%M');
name_id = "_reg_path_comp"
run_id = time_id + name_id
run_id_path = os.path.join("output", run_id)
if not os.path.exists(run_id_path):
    os.makedirs(run_id_path)
    os.makedirs(os.path.join(run_id_path, "positive_weights"))

num_samples = 500
noise_scale = 1
beta = np.array([1, -1.2, 1.5, -0.3, 0.5])
num_features = len(beta)
epochs = 1000

# Torch params
learning_rate = 0.002
np.random.seed(1)
torch.manual_seed(1)
X, y = create.regression_example_data(num_samples=num_samples,
                                      num_features=num_features,
                                      noise_scale=noise_scale,
                                      beta=beta)


regularizers = np.array(["stew", "stem2", "stow", "stnw", "sted"])
regularizer_names = np.array(["Shrinkage toward equal weights",
                              "Shrinkage toward equal weight magnitudes",
                              "Shrinkage toward ordered weights",
                              "Shrinkage toward noncompensatory weights",
                              "Shrinkage toward exponentially decaying weights"])
# regularizers = np.array(["sted"])
# regularizer_names = np.array(["Shrinkage toward exponentially decaying weights"])
num_regularizers = len(regularizers)
lambda_min = -3
lambda_max = 2
num_lambdas = 40
# lams = np.insert(np.logspace(lambda_min, lambda_max, num=num_lambdas-1), 0, 0.0)
lams = np.logspace(lambda_min, lambda_max, num=num_lambdas)

weight_storage = np.zeros((num_regularizers, num_lambdas, num_features))

for reg_ix, regularizer in enumerate(regularizers):  # regularizer = "stew"
    print(reg_ix, regularizer)
    for lam_ix, lam in enumerate(lams):  # lam = 0.1
        lin_reg_torch = LinearRegressionTorch(num_features=num_features,
                                              learning_rate=learning_rate,
                                              regularization=regularizer,
                                              lam=lam)
        betas = lin_reg_torch.fit(X, y, epochs=epochs).detach().numpy()
        weight_storage[reg_ix, lam_ix] = betas


# Plot
for reg_ix, regularizer in enumerate(regularizers):
    fig1, ax1 = plt.subplots()
    plt.title(regularizer_names[reg_ix])
    for weight_ix in range(num_features):
        ax1.plot(lams, weight_storage[reg_ix, :, weight_ix], label="beta_" + str(weight_ix+1))
    plt.xscale("log")
    plt.axhline(y=0, color="grey")
    plt.legend()
    fig1.show()
    fig1.savefig(os.path.join(run_id_path, regularizer))
    plt.close()



## On positive weights


beta = np.array([1, 0.8, 1.5, 0.3, 0.5])
X, y = create.regression_example_data(num_samples=num_samples,
                                      num_features=num_features,
                                      noise_scale=noise_scale,
                                      beta=beta)

weight_storage = np.zeros((num_regularizers, num_lambdas, num_features))

for reg_ix, regularizer in enumerate(regularizers):  # regularizer = "stew"
    print(reg_ix, regularizer)
    for lam_ix, lam in enumerate(lams):  # lam = 0.1
        lin_reg_torch = LinearRegressionTorch(num_features=num_features,
                                              learning_rate=learning_rate,
                                              regularization=regularizer,
                                              positivity_constraint=True,
                                              lam=lam)
        betas = lin_reg_torch.fit(X, y, epochs=epochs).detach().numpy()
        weight_storage[reg_ix, lam_ix] = betas


# Plot
for reg_ix, regularizer in enumerate(regularizers):
    fig1, ax1 = plt.subplots()
    plt.title(regularizer_names[reg_ix])
    for weight_ix in range(num_features):
        ax1.plot(lams, weight_storage[reg_ix, :, weight_ix], label="beta_" + str(weight_ix+1))
    plt.xscale("log")
    plt.axhline(y=0, color="grey")
    plt.legend()
    fig1.show()
    fig1.savefig(os.path.join(run_id_path, "positive_weights", regularizer))
    plt.close()

