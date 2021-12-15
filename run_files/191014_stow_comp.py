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
name_id = "_stow_comp"
run_id = time_id + name_id
run_id_path = os.path.join("/Users/malte/Dropbox/projects/ozgur/shrinkage toward noncompensatory weights/figures/", run_id)
if not os.path.exists(run_id_path):
    os.makedirs(run_id_path)
    # os.makedirs(os.path.join(run_id_path, "positive_weights"))

num_samples = 40
test_set_size = 1000
noise_scale = 3
beta = np.array([1, 0.8, 1.2, 0.4, 0.6])
num_features = len(beta)
epochs = 1000
verbose = False

# Torch params
learning_rates = [0.001, 0.002, 0.005]
np.random.seed(1)
torch.manual_seed(1)
X, y = create.regression_example_data(num_samples=num_samples,
                                      num_features=num_features,
                                      noise_scale=noise_scale,
                                      beta=beta)
X_test, y_test = create.regression_example_data(num_samples=test_set_size,
                                                num_features=num_features,
                                                noise_scale=noise_scale,
                                                beta=beta)

regularizers = np.array(["stow", "stnw", "sted", "sthd"])  # , "stow", "stnw", "sted"
regularizer_names = np.array(["Shrinkage toward ordered weights",
                              "Shrinkage toward noncompensatory weights",
                              "Shrinkage toward exponentially decaying weights",
                              "Shrinkage toward harmonically decaying weights"])

regularizer_labels = np.array(["Shrinkage toward\nordered weights",
                              "Shrinkage toward\nnoncompensatory weights",
                              "Shrinkage toward\nexponentially decaying weights",
                              "Shrinkage toward\nharmonically decaying weights"])

# regularizers = np.array(["sted"])
# regularizer_names = np.array(["Shrinkage toward exponentially decaying weights"])
num_regularizers = len(regularizers)
lambda_min = -3
lambda_max = 2
num_lambdas = 40
# lams = np.insert(np.logspace(lambda_min, lambda_max, num=num_lambdas-1), 0, 0.0)
lams = np.logspace(lambda_min, lambda_max, num=num_lambdas)


for lr_ix, lr in enumerate(learning_rates):
    weight_storage = np.zeros((num_regularizers, num_lambdas, num_features))
    errors = np.zeros((num_regularizers, num_lambdas))
    for reg_ix, regularizer in enumerate(regularizers):  # regularizer = "stew"
        print(reg_ix, regularizer)
        for lam_ix, lam in enumerate(lams):  # lam = 0.1
            lin_reg_torch = LinearRegressionTorch(num_features=num_features,
                                                  learning_rate=learning_rates[lr_ix],
                                                  regularization=regularizer,
                                                  lam=lam,
                                                  verbose=verbose)
            betas = lin_reg_torch.fit(X, y, epochs=epochs).detach().numpy()
            y_pred = lin_reg_torch.predict(X_test).numpy().flatten()
            error = np.mean(np.power(y_test - y_pred, 2))
            errors[reg_ix, lam_ix] = error
            weight_storage[reg_ix, lam_ix] = betas


    # Plot
    for reg_ix, regularizer in enumerate(regularizers):
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        plt.title(regularizer_names[reg_ix])
        for weight_ix in range(num_features):
            ax1.plot(lams, weight_storage[reg_ix, :, weight_ix], label="beta_" + str(weight_ix+1))
        plt.xscale("log")
        plt.axhline(y=0, color="grey")
        plt.legend()
        fig1.show()
        fig1.savefig(os.path.join(run_id_path, regularizer + str(lr_ix) + ".pdf"))
        plt.close()

    # Plot error paths
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.title("Errors")
    plt.xlabel("Regularization strength")
    plt.ylabel("Mean squared error")
    plt.xscale("log")
    for reg_ix, regularizer in enumerate(regularizers):
        ax1.plot(lams, errors[reg_ix, :], label=regularizer_labels[reg_ix])
    # plt.legend()
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
    fig1.show()
    fig1.savefig(os.path.join(run_id_path, "errors" + str(lr_ix) + ".pdf"))
    plt.close()

