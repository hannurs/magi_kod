from FeatureSelectionMethod import FeatureSelectionMethod
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector

class ForwardSelection(FeatureSelectionMethod):
    @classmethod
    def saveKBestFeatures(self, k, data_filename):
        predictors, Xdata, Ydata = super().getTrainData(data_filename)
        features = []
        knn = KNeighborsClassifier(n_neighbors=3)
        selector = SequentialFeatureSelector(knn, n_features_to_select=k)
        selector.fit(Xdata, Ydata)
        features = selector.get_feature_names_out(predictors[:-1])

        self.saveToPickle(features, k, data_filename)

class BackwardSelection(FeatureSelectionMethod):
    @classmethod
    def saveKBestFeatures(self, k, data_filename):
        predictors, Xdata, Ydata = super().getTrainData(data_filename)
        features = []
        knn = KNeighborsClassifier(n_neighbors=3)
        selector = SequentialFeatureSelector(knn, n_features_to_select=k, direction="backward")
        selector.fit(Xdata, Ydata)
        features = selector.get_feature_names_out(predictors[:-1])

        self.saveToPickle(features, k, data_filename)