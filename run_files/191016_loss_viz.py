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
import seaborn as sns; sns.set()
# from stew.regression import LinearRegressionTorch
# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)


time_id = datetime.now().strftime('%Y_%m_%d_%H_%M');
name_id = "_loss_viz"
run_id = time_id + name_id
run_id_path = os.path.join("/Users/malte/Dropbox/projects/ozgur/stew journal/figures/", run_id)
if not os.path.exists(run_id_path):
    os.makedirs(run_id_path)
    # os.makedirs(os.path.join(run_id_path, "positive_weights"))


def stew_penalty(b1, b2, q):
    return np.power(np.abs(b1 - b2), q)


def stem_penalty(b1, b2, q):
    return np.power(np.abs(np.abs(b1) - np.abs(b2)), q)

figsize = (4, 3)
num_values = 100
bound = 2

y, x = np.meshgrid(np.linspace(-bound, bound, num_values),
                   np.linspace(-bound, bound, num_values))
z = stew_penalty(x, y, 1)
z_min, z_max = 0, np.abs(z).max()
fig, ax = plt.subplots(figsize=figsize)
c = ax.pcolormesh(x, y, z, cmap='coolwarm', vmin=z_min, vmax=z_max)
# ax.set_title('STEW (q=1)')
ax.axis([x.min(), x.max(), y.min(), y.max()])
# plt.title("STEW (q=1) in two dimensions")
plt.xlabel(r"$\beta_1$")
plt.ylabel(r"$\beta_2$")
cb = plt.colorbar(c, ax=ax)
cb.set_label(r"STEW penalty $|\beta_1 - \beta_2|$")
# plt.legend()
plt.show()
fig.savefig(os.path.join(run_id_path, "stew1.pdf"))


y, x = np.meshgrid(np.linspace(-bound, bound, num_values),
                   np.linspace(-bound, bound, num_values))
z = stew_penalty(x, y, 2)
z_min, z_max = 0, np.abs(z).max()
fig, ax = plt.subplots(figsize=figsize)
c = ax.pcolormesh(x, y, z, cmap='coolwarm', vmin=z_min, vmax=z_max)
# ax.set_title('STEW (q=2)')
ax.axis([x.min(), x.max(), y.min(), y.max()])
plt.xlabel(r"$\beta_1$")
plt.ylabel(r"$\beta_2$")
cb = plt.colorbar(c, ax=ax)
cb.set_label(r"STEW penalty $|\beta_1 - \beta_2|$")
plt.show()
fig.savefig(os.path.join(run_id_path, "stew2.pdf"))


y, x = np.meshgrid(np.linspace(-bound, bound, num_values),
                   np.linspace(-bound, bound, num_values))
z = stem_penalty(x, y, 1)
z_min, z_max = 0, np.abs(z).max()
fig, ax = plt.subplots(figsize=figsize)
c = ax.pcolormesh(x, y, z, cmap='coolwarm', vmin=z_min, vmax=z_max)
# ax.set_title('STEM (q=1)')
ax.axis([x.min(), x.max(), y.min(), y.max()])
plt.xlabel(r"$\beta_1$")
plt.ylabel(r"$\beta_2$")
cb = plt.colorbar(c, ax=ax)
cb.set_label(r"STEM penalty $|\ |\beta_1| - |\beta_2|\ |$")
plt.show()
fig.savefig(os.path.join(run_id_path, "stem1.pdf"))


y, x = np.meshgrid(np.linspace(-bound, bound, num_values),
                   np.linspace(-bound, bound, num_values))
z = stem_penalty(x, y, 2)
z_min, z_max = 0, np.abs(z).max()
fig, ax = plt.subplots(figsize=figsize)
c = ax.pcolormesh(x, y, z, cmap='coolwarm', vmin=z_min, vmax=z_max)
# ax.set_title('STEM (q=2)')
ax.axis([x.min(), x.max(), y.min(), y.max()])
plt.xlabel(r"$\beta_1$")
plt.ylabel(r"$\beta_2$")
cb = plt.colorbar(c, ax=ax)
cb.set_label(r"STEM penalty $|\ |\beta_1| - |\beta_2|\ |$")
plt.show()
fig.savefig(os.path.join(run_id_path, "stem2.pdf"))



y, x = np.meshgrid(np.linspace(-bound, bound, num_values),
                   np.linspace(-bound, bound, num_values))
z = stew_penalty(x, y, 1) - stem_penalty(x, y, 1)
z_min, z_max = 0, np.abs(z).max()
fig, ax = plt.subplots(figsize=figsize)
c = ax.pcolormesh(x, y, z, cmap='coolwarm', vmin=z_min, vmax=z_max)
# ax.set_title('STEM (q=2)')
ax.axis([x.min(), x.max(), y.min(), y.max()])
plt.xlabel(r"$\beta_1$")
plt.ylabel(r"$\beta_2$")
cb = plt.colorbar(c, ax=ax)
cb.set_label(r"STEW - STEM")
plt.show()
fig.savefig(os.path.join(run_id_path, "stewMstem.pdf"))




#
#
#
#
#
# betas_1 = np.linspace(-3, 3, num=num_values)
# betas_2s = np.linspace(-3, 3, num=num_values)
# stews_1 = np.zeros((num_values, num_values))
# stews_2 = np.zeros((num_values, num_values))
# stems_1 = np.zeros((num_values, num_values))
# stems_2 = np.zeros((num_values, num_values))
#
# for b1_ix, b1 in enumerate(betas_1):
#     for b2_ix, b2 in enumerate(betas_2):
#         stews_1[b1_ix, b2_ix] = stew_penalty(b1, b2, 1)
#         stews_2[b1_ix, b2_ix] = stew_penalty(b1, b2, 2)
#         stems_1[b1_ix, b2_ix] = stem_penalty(b1, b2, 1)
#         stems_2[b1_ix, b2_ix] = stem_penalty(b1, b2, 2)
#
#
# ax = sns.heatmap(stews_1)
# plt.show()
# ax = sns.heatmap(stews_2)
# plt.show()
# ax = sns.heatmap(stems_1)
# plt.show()
# ax = sns.heatmap(stems_2)
# plt.show()
#
#
#
# num_samples = 40
# test_set_size = 1000
# noise_scale = 3
# beta = np.array([1, -1.2, 1.5, -0.3, 0.5])
# num_features = len(beta)
# epochs = 1000
# verbose = False
#
# # Torch params
# learning_rates = [0.001, 0.002, 0.005]
# np.random.seed(1)
# torch.manual_seed(1)
# X, y = create.regression_example_data(num_samples=num_samples,
#                                       num_features=num_features,
#                                       noise_scale=noise_scale,
#                                       beta=beta)
# X_test, y_test = create.regression_example_data(num_samples=test_set_size,
#                                                 num_features=num_features,
#                                                 noise_scale=noise_scale,
#                                                 beta=beta)
#
# regularizers = np.array(["stem_opt",
#                          "ridge",
#                          "lasso",
#                          "stew1",
#                          "stew2",
#                          "stem1",
#                          "stem2"])  # , "stow", "stnw", "sted"
# regularizer_names = np.array(["Optimized STEM",
#                               "Ridge",
#                               "Lasso",
#                               "STEW: Shrinkage toward equal weights (q=1)",
#                               "STEW: Shrinkage toward equal weights (q=2)",
#                               "STEM: Shrinkage toward equal magnitudes (q=1)",
#                               "STEM: Shrinkage toward equal magnitudes (q=2)"])
# regularizer_labels = np.array(["Optimized STEM",
#                                "Ridge",
#                                "Lasso",
#                                "STEW: Shrinkage toward \nequal weights (q=1)",
#                                "STEW: Shrinkage toward \nequal weights (q=2)",
#                                "STEM: Shrinkage toward \nequal magnitudes (q=1)",
#                                "STEM: Shrinkage toward \nequal magnitudes (q=2)"])
#
#
# # regularizers = np.array(["stem_opt"])
# # regularizer_names = np.array(["Optimized STEM"])
# # regularizer_labels = np.array(["Optimized STEM"])
#
# num_regularizers = len(regularizers)
# lambda_min = -3
# lambda_max = 3
# num_lambdas = 40
# # lams = np.insert(np.logspace(lambda_min, lambda_max, num=num_lambdas-1), 0, 0.0)
# lams = np.logspace(lambda_min, lambda_max, num=num_lambdas)
#
#
# for lr_ix, lr in enumerate(learning_rates):  # lr_ix = 0; lr = 0.001
#     weight_storage = np.zeros((num_regularizers, num_lambdas, num_features))
#     errors = np.zeros((num_regularizers, num_lambdas))
#     for reg_ix, regularizer in enumerate(regularizers):  # regularizer = "stew"
#         print(reg_ix, regularizer)
#         for lam_ix, lam in enumerate(lams):  # lam = 0.1
#             if regularizer == "stem_opt":
#                 lin_reg = STEMopt(train_fraction=0.8,
#                                   num_features=num_features,
#                                   learning_rate=learning_rates[lr_ix],
#                                   regularization=regularizer,
#                                   lam=lam,
#                                   verbose=verbose)
#             else:
#                 lin_reg = LinearRegressionTorch(num_features=num_features,
#                                                 learning_rate=learning_rates[lr_ix],
#                                                 regularization=regularizer,
#                                                 lam=lam,
#                                                 verbose=verbose)
#             betas = lin_reg.fit(X, y, epochs=epochs)  # .detach().numpy()
#             y_pred = lin_reg.predict(X_test)
#             error = np.mean(np.power(y_test - y_pred, 2))
#             errors[reg_ix, lam_ix] = error
#             weight_storage[reg_ix, lam_ix] = betas
#
#     # Plot reg paths
#     for reg_ix, regularizer in enumerate(regularizers):
#         fig1, ax1 = plt.subplots(figsize=(8, 6))
#         plt.title(regularizer_names[reg_ix])
#         for weight_ix in range(num_features):
#             ax1.plot(lams, weight_storage[reg_ix, :, weight_ix], label="beta_" + str(weight_ix+1))
#         plt.xscale("log")
#         plt.axhline(y=0, color="grey")
#         plt.legend()
#         fig1.show()
#         fig1.savefig(os.path.join(run_id_path, regularizer + str(lr_ix) + ".pdf"))
#         plt.close()
#
#     # Plot error paths
#     fig1, ax1 = plt.subplots(figsize=(10, 6))
#     ax1.spines['top'].set_visible(False)
#     ax1.spines['right'].set_visible(False)
#     plt.title("Errors")
#     plt.xlabel("Regularization strength")
#     plt.ylabel("Mean squared error")
#     plt.xscale("log")
#     for reg_ix, regularizer in enumerate(regularizers):
#         ax1.plot(lams, errors[reg_ix, :], label=regularizer_labels[reg_ix])
#     # plt.legend()
#     plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
#     fig1.show()
#     fig1.savefig(os.path.join(run_id_path, "errors" + str(lr_ix) + ".pdf"))
#     plt.close()
#
#
# # ## On positive weights
# #
# #
# # beta = np.array([1, 0.8, 1.5, 0.3, 0.5])
# # X, y = create.regression_example_data(num_samples=num_samples,
# #                                       num_features=num_features,
# #                                       noise_scale=noise_scale,
# #                                       beta=beta)
# #
# # weight_storage = np.zeros((num_regularizers, num_lambdas, num_features))
# #
# # for reg_ix, regularizer in enumerate(regularizers):  # regularizer = "stew"
# #     print(reg_ix, regularizer)
# #     for lam_ix, lam in enumerate(lams):  # lam = 0.1
# #         lin_reg = LinearRegressionTorch(num_features=num_features,
# #                                               learning_rate=learning_rate,
# #                                               regularization=regularizer,
# #                                               positivity_constraint=True,
# #                                               lam=lam)
# #         betas = lin_reg.fit(X, y, epochs=epochs).detach().numpy()
# #         weight_storage[reg_ix, lam_ix] = betas
# #
# #
# # # Plot
# # for reg_ix, regularizer in enumerate(regularizers):
# #     fig1, ax1 = plt.subplots(figsize=(5, 3.5))
# #     plt.title(regularizer_names[reg_ix])
# #     for weight_ix in range(num_features):
# #         ax1.plot(lams, weight_storage[reg_ix, :, weight_ix], label="beta_" + str(weight_ix+1))
# #     plt.xscale("log")
# #     plt.axhline(y=0, color="grey")
# #     plt.legend()
# #     fig1.show()
# #     fig1.savefig(os.path.join(run_id_path, "positive_weights", regularizer + ".pdf"))
# #     plt.close()
# #
