import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


X = np.load('make_data/dataset/training_data.npy') #we want in shape (itter,kbins)
y = pd.read_csv('make_data/dataset/training_labels.csv')  
y = y['f* 10']
print(X.shape)
print(y.shape)
# Split data into train partition and test partition
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.7)

mlp = MLPRegressor(
    hidden_layer_sizes=(40,40,40),
    max_iter=8,
    alpha=1e-4,
    solver="adam",
    verbose=10,
    random_state=1,
    learning_rate_init=0.001,
    activation='relu',
)

# this example won't converge because of resource usage constraints on
# our Continuous Integration infrastructure, so we catch the warning and
# ignore it here
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))