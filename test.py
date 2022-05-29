import pickle
from random import betavariate
import numpy as np
import os
import csv
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# f = open("scores/filter/reliefcsv_result-easyReliefF.pkl", "rb")
# output = pickle.load(f)
# print(output)
def read_file(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        predictors = list(reader)[0]
    
    alldata = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
    Xdata = alldata[:, :-1]
    Ydata = alldata[:, -1]

    return predictors, alldata


# i = 0
# for filename in os.listdir("databases/extracted/test"):
    
#     predictors, alldata = read_file("databases/extracted/test/" + filename)

#     if predictors[0] == "id":
#         alldata = np.delete(alldata, 0, 1)
#         predictors = predictors[1:]

#         with open("databases/extracted/test/" + filename, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(predictors)
#             writer.writerows(alldata)
#     i+=1

# i = 0
# for filename in os.listdir("databases/extracted/train"):
    
#     predictors, alldata = read_file("databases/extracted/train/" + filename)

#     if predictors[0] == "id":
#         alldata = np.delete(alldata, 0, 1)
#         predictors = predictors[1:]

#         with open("databases/extracted/train/" + filename, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(predictors)
#             writer.writerows(alldata)
#     i+=1

# for i in range(4):
#     f = open("scores/wrapper/forward_selection/" + os.listdir("scores/wrapper/forward_selection")[i], "rb")
#     output = pickle.load(f)
#     print(output)
#     print()


import statsmodels.api as sm
from sklearn.datasets import load_boston

data = load_boston()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=False):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = excluded[new_pval.argmin()]
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = included[pvalues.argmax()]
            print(worst_feature)
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

# print(X)
# # print(y)
# result = stepwise_selection(X, y)

# print('resulting features:')
# print(result)