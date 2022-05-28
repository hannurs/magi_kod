import numpy as np
import csv
import sklearn
import sklearn.feature_selection
import sklearn_relief
from skrebate import ReliefF
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os
import pickle
from enum import Enum
import minepy

class FilterFSMethod(Enum):
    ReliefF = 0
    MutualInformation = 1
    MaximalInformationCoefficient = 2
    ANOVA = 3

def read_file(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        predictors = list(reader)[0]
    
    alldata = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
    Xdata = alldata[:, :-1]
    Ydata = alldata[:, -1]

    return predictors, preprocessing.normalize(Xdata), Ydata

def ROCAUCpredictors(n_sel_pred, predictors, Xdata, Ydata):
    print()

def measureFeatureScores(method):
    dir_path = "databases/extracted/train"
    for filename in os.listdir(dir_path):
        predictors, Xdata, Ydata = read_file(dir_path + "/" + filename)
        feature_scores = []
        fi_dict = {}
        if method == FilterFSMethod.ReliefF:
            fs = ReliefF()
            fs.fit(Xdata, Ydata)
            feature_scores = fs.feature_importances_
            pickle_path = "scores/filter/relief/"
            pickle_name = "ReliefF.pkl"
        elif method == FilterFSMethod.MutualInformation:
            feature_scores = sklearn.feature_selection.mutual_info_classif(Xdata, Ydata)
            pickle_path = "scores/filter/mutual_information/"
            pickle_name = "MutualInformation.pkl"
        elif method == FilterFSMethod.MaximalInformationCoefficient:
            Ydata2D = np.reshape(Ydata, (len(Ydata), 1))
            # print(Ydata2D.shape)
            # print(Xdata.shape)
            mic, tic = minepy.cstats(Xdata.T, Ydata2D.T)
            for score in mic:
                feature_scores.append(score[0])
            pickle_path = "scores/filter/maximal_information_coefficient/"
            pickle_name = "MaximalInformationCoefficient.pkl"
        elif method == FilterFSMethod.ANOVA:
            pickle_path = "scores/filter/t-test/"
            pickle_name = "TTest.pkl"
            selector = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.f_classif, k=5)
            selector.fit(Xdata, Ydata)
            feature_scores = selector.scores_
            best = selector.get_support(indices=True)
            for k in best:
                print(predictors[k], ": ", feature_scores[k])
            # f, feature_scores = sklearn.feature_selection.f_classif(Xdata, Ydata)
            # print(feature_scores)
        
        i = 0
        print(filename)
        for f_score in feature_scores:
            # print(predictors[i], '\t', f_score)
            fi_dict[predictors[i]] = f_score
            i+=1
        with open(pickle_path + filename[:-9] + pickle_name, "wb") as f:
            pickle.dump(fi_dict, f)


##### MEASURE FEATURE PERFORMANCE WITH RELIEFF AND SAVE
# filenames = []
# dir_path = "databases/extracted/train"
# for filename in os.listdir(dir_path):
#     filenames.append(filename)
#     print(filename[:-9])

#     predictors, Xdata, Ydata = read_file(dir_path + "/" + filename)
#     fs = ReliefF()
#     fs.fit(Xdata, Ydata)
#     fi_dict = {}
#     i = 0
#     for f_score in fs.feature_importances_:
#         print(predictors[i], '\t', f_score)
#         fi_dict[predictors[i]] = f_score
#         i+=1
#     with open("scores/filter/relief/" + filename[:-9] + "ReliefF.pkl", "wb") as f:
#         pickle.dump(fi_dict, f)
    # plt.bar(predictors, fs.feature_importances_)
#####

# measureFeatureScores(FilterFSMethod.MutualInformation)
# measureFeatureScores(FilterFSMethod.MaximalInformationCoefficient)
# measureFeatureScores(FilterFSMethod.ReliefF)
measureFeatureScores(FilterFSMethod.ANOVA)