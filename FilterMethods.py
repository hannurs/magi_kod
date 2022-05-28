from sklearn.feature_selection import f_classif, SelectKBest
from FeatureSelectionMethod import FeatureSelectionMethod

class ANOVA(FeatureSelectionMethod):

    @classmethod
    def saveKBestFeatures(self, k, data_filename):
        # print("ok")
        print(self.__name__)
        selector = SelectKBest(f_classif, k=k)
        predictors, Xdata, Ydata = super().getTrainData(data_filename)
        selector.fit(Xdata, Ydata)
        feature_scores = selector.scores_
        best_features = selector.get_support(indices=True)

        fi_dict = {}
        for i in best_features:
            fi_dict[predictors[i]] = feature_scores[i]

        self.saveToPickle(fi_dict, k, data_filename)