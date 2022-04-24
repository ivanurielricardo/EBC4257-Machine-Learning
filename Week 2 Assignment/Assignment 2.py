# Example of scenarios 1 and 2 as described in Elements of Statistical Learning II (page 13), no warranty.
# Course: EBC4257 - Machine Learning, SBE, UM
# Author: RJAlmeida, April 2021

import numpy as np
import pandas as pd
import matplotlib as plt
import sklearn as sk
import random

import matplotlib.image as pltimg
# matplotlib.use('TkAgg')  # to open pictures in new window, currently saving them
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics





mu1 = None
mu2 = None
n_data_points = 1000
plot_graphs = 1



def scenarios_generation(mu1=None, mu2=None, n_data_points=1000, plot_graphs=1):
    if mu1 is None:
        mu1 = [0, 1]
    if mu2 is None:
        mu2 = [2, 3]

    np.random.seed(42)
    xa = np.random.multivariate_normal(mu1, np.eye(2), n_data_points)
    ya = np.zeros(n_data_points)

    xb = np.random.multivariate_normal(mu2, np.eye(2), n_data_points)
    yb = np.ones(n_data_points)

    data_x = np.concatenate((xa, xb))
    data_y = np.concatenate((ya, yb)).reshape(-1, 1)

    data_scenario1 = pd.DataFrame(data=np.concatenate((data_x, data_y), 1), columns=['X1', 'X2', 'Y'])

    groups = data_scenario1.groupby('Y')

    if plot_graphs == 1:
        fig, ax = plt.subplots()
        for name, group in groups:
            ax.plot(group['X1'], group['X2'], marker='o', linestyle='', alpha=.5, label=name)
        ax.legend(loc="upper left")
        plt.savefig("Scenario1.png")
        plt.show()

    n_groups = 10
    data_x2 = np.empty((0, 2))
    mean_all = np.random.multivariate_normal(mu1, np.eye(2), n_groups)
    for mu in mean_all:
        data_x2 = np.vstack(
            [data_x2, np.random.multivariate_normal(mu, 0.1 * np.eye(2), int(round(n_data_points / 10)))])

    mean_all = np.random.multivariate_normal(mu2, np.eye(2), n_groups)
    for mu in mean_all:
        data_x2 = np.vstack(
            [data_x2, np.random.multivariate_normal(mu, 0.1 * np.eye(2), int(round(n_data_points / 10)))])

    data_scenario2 = pd.DataFrame(data=np.concatenate((data_x2, data_y), 1), columns=['X1', 'X2', 'Y'])

    groups = data_scenario2.groupby('Y')

    if plot_graphs == 1:
        fig, ax = plt.subplots()
        for name, group in groups:
            ax.plot(group['X1'], group['X2'], marker='o', linestyle='', alpha=.5, label=name)
        ax.legend(loc="upper left")
        plt.savefig("Scenario2.png")
        plt.show()


if __name__ == '__main__':
    scenarios_generation()

# Now we have the data saved as a data frame
# Can now create a decision tree

scenario1_features = pd.concat([data_scenario1.X1, data_scenario1.X2], join = 'outer', axis = 1)

# BOOTSTRAP Resampling

scenario1_bootstrap = data_scenario1.sample(len(data_scenario1), replace = True)

# We create a decision tree using the bootstrapped dataset

dtree = DecisionTreeClassifier()
dtree = dtree.fit(scenario1_bootstrap[["X1", "X2"]], scenario1_bootstrap[["Y"]])



X_train, X_test, y_train, y_test = train_test_split(data_scenario1.X, data_y, test_size = 0.10)

# Create Gaussian Classifier
clf = RandomForestClassifier(n_estimators=1)
clf.fit(X_train, np.ravel(y_train))  # change from column to row
y_pred = clf.predict(X_test)

# We can now check the accuracy of the training and testing set
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


