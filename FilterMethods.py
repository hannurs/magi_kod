from sklearn.feature_selection import f_classif, SelectKBest, mutual_info_classif
from FeatureSelectionMethod import FeatureSelectionMethod
import minepy
import numpy as np
import skrebate

class ANOVA(FeatureSelectionMethod):
    filterMethod = True

    @classmethod
    def saveKBestFeatures(self, k, data_filename):
        selector = SelectKBest(f_classif, k=k)
        predictors, Xdata, Ydata = super().getTrainData(data_filename)
        selector.fit(Xdata, Ydata)
        feature_scores = selector.scores_
        best_features = selector.get_support(indices=True)

        fi_dict = {}
        for i in best_features:
            fi_dict[predictors[i]] = feature_scores[i]

        self.saveToPickle(fi_dict, k, data_filename)

class MutualInformation(FeatureSelectionMethod):
    filterMethod = True
    
    @classmethod
    def saveKBestFeatures(self, k, data_filename):
        selector = SelectKBest(mutual_info_classif, k=k)
        predictors, Xdata, Ydata = super().getTrainData(data_filename)
        selector.fit(Xdata, Ydata)
        feature_scores = selector.scores_
        best_features = selector.get_support(indices=True)

        fi_dict = {}
        for i in best_features:
            fi_dict[predictors[i]] = feature_scores[i]

        self.saveToPickle(fi_dict, k, data_filename)

class MaximalInformationCoefficient(FeatureSelectionMethod):
    filterMethod = True
    
    @classmethod
    def saveKBestFeatures(self, k, data_filename):
        predictors, Xdata, Ydata = super().getTrainData(data_filename)
        Ydata2D = np.reshape(Ydata, (len(Ydata), 1))
        mic, tic = minepy.cstats(Xdata.T, Ydata2D.T)

        feature_scores = []
        for mic_score in mic:
            feature_scores.append(mic_score[0])

        fi_dict = {}
        best_features = np.argpartition(feature_scores, -k)[-k:]
        for i in best_features:
            fi_dict[predictors[i]] = feature_scores[i]

        self.saveToPickle(fi_dict, k, data_filename)

        
class ReliefF(FeatureSelectionMethod):
    filterMethod = True
    
    @classmethod
    def saveKBestFeatures(self, k, data_filename):
        predictors, Xdata, Ydata = super().getTrainData(data_filename)
        selector = skrebate.ReliefF()
        selector.fit(Xdata, Ydata)
        feature_scores = selector.feature_importances_

        fi_dict = {}
        best_features = np.argpartition(feature_scores, -k)[-k:]
        for i in best_features:
            fi_dict[predictors[i]] = feature_scores[i]

        self.saveToPickle(fi_dict, k, data_filename)