import numpy as np
import pandas as pd
import matplotlib as plt
import sklearn
import seaborn as sns

import matplotlib.image as pltimg
# matplotlib.use('TkAgg')  # to open pictures in new window, currently saving them
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.ensemble import RandomForestRegressor




mu1 = [0,1]
mu2 = [2,3]
n_data_points = 1000
plot_graphs = 1

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

# BOOTSTRAP Resampling

scenario1_bootstrap = data_scenario1.sample(len(data_scenario1), replace = True)

# We create a decision tree using the bootstrapped dataset

dtree = DecisionTreeClassifier()
dtree = dtree.fit(scenario1_bootstrap[["X1", "X2"]], scenario1_bootstrap[["Y"]])

# Training and Testing set
X_train, X_test, y_train, y_test = train_test_split(data_scenario1[["X1", "X2"]], data_scenario1[["Y"]], test_size = 0.3)


############### Using a Random Forest ###############

clf = RandomForestClassifier(n_estimators = 2)
clf = clf.fit(X_train, np.ravel(y_train))



clf_lab = clf.predict(X_train)
clf_pre = clf.predict(X_test)
print("Oob Score:", clf.oob_score_)
print("The precision on the training data set:", accuracy_score(y_train, clf_lab))
print("Verify the accuracy on the data set:", accuracy_score(y_test, clf_pre))

y_pred = clf.predict(X_test)
df_y = pd.DataFrame(y_pred)

MSE = (np.subtract(df_y,y_test))**2

MSE.mean()



############### Giving the ROC curves ###############

pre_y = clf.predict_proba(X_test)[:, 1]
fpr_Nb, tpr_Nb, _ = roc_curve(y_test, pre_y)
aucval = auc(fpr_Nb, tpr_Nb) # calculates the value of AUC
plt.figure(figsize=(10,8))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_Nb, tpr_Nb,"r",linewidth = 3)
plt.grid()
plt.xlabel ("false positive rate")
plt.ylabel ("Real Rate")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title ("Random Forest ROC Curve")
plt.text(0.15,0.9,"AUC = "+str(round(aucval,4)))
plt.show()

############### Giving the Learning Curve ###############

def learning_curves(estimator, data, features, target, train_sizes, cv):
    train_sizes, train_scores, validation_scores = learning_curve(
    estimator, data[features], data[target], train_sizes = train_sizes,
    cv = cv, scoring = 'neg_mean_squared_error')
    train_scores_mean = -train_scores.mean(axis = 1)
    validation_scores_mean = -validation_scores.mean(axis = 1)

    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
    plt.ylabel('MSE', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    title = 'Learning curves for a ' + str(estimator).split('(')[0] + ' model'
    plt.title(title, fontsize = 18, y = 1.03)
    plt.legend()
    plt.ylim(0,40)


plt.figure(figsize = (16,5))

learning_curves(clf, )