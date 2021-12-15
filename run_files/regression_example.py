import numpy as np
import stew.example_data as create
import stew.mlogit as mlogit
import stew.utils as utils
import matplotlib.pyplot as plt
from stew.utils import create_diff_matrix
from sklearn.linear_model import LinearRegression
from stew.regression import *
# from stew.regression import LinearRegressionTorch

num_samples = 600
noise_scale = 1
beta = np.array([1, 2, 1.5, 1.8])
num_features = len(beta)
epochs = 1000

# Torch params
learning_rate = 0.005
np.random.seed(1)
torch.manual_seed(1)
X, y = create.regression_example_data(num_samples=num_samples,
                                      num_features=num_features,
                                      noise_scale=noise_scale,
                                      beta=beta)



# # PyTorch stnw (= noncompensatory weights)
lin_reg_torch = LinearRegressionTorch(num_features=num_features, learning_rate=learning_rate, regularization="stnw", lam=5)
lin_reg_torch.fit(X, y, epochs=epochs)
lin_reg_torch.model.input_linear.weight


# # PyTorch STOW (= ordered weights)
lin_reg_torch = LinearRegressionTorch(num_features=num_features, learning_rate=learning_rate, regularization="stow", lam=5)
lin_reg_torch.fit(X, y, epochs=epochs)
lin_reg_torch.model.input_linear.weight


# # PyTorch STEM (= equal magnitudes)
lin_reg_torch = LinearRegressionTorch(num_features=num_features, learning_rate=learning_rate, regularization="stem", lam=5)
lin_reg_torch.fit(X, y, epochs=epochs)
lin_reg_torch.model.input_linear.weight


# # PyTorch STEW
lin_reg_torch = LinearRegressionTorch(num_features=num_features, learning_rate=learning_rate, regularization="stew", lam=2)
lin_reg_torch.fit(X, y, epochs=epochs)
lin_reg_torch.model.input_linear.weight

# # OLS
reg = LinearRegression(fit_intercept=False).fit(X=X, y=y)
reg.score(X, y)
print("OLS estimates: ", reg.coef_)


# # STEW
D = create_diff_matrix(num_features=num_features)
lam = 0
print(f"STEW with lambda = {lam}: {stew_reg(X, y, D, lam=lam)}")

lam = 10000
print(f"STEW with lambda = {lam}: {stew_reg(X, y, D, lam=lam)}")





