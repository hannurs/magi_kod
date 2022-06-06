

from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from FeatureSelectionMethod import FeatureSelectionMethod


class RecursiveFeatureElimination(FeatureSelectionMethod):
    @classmethod
    def saveKBestFeatures(self, k, data_filename):
        predictors, Xdata, Ydata = super().getTrainData(data_filename)
        features = []
        svm_ = SVC(kernel="linear")
        knn = KNeighborsClassifier(n_neighbors=3)
        
        eliminator = RFE(svm_, n_features_to_select=k)
        eliminator.fit(Xdata, Ydata)
        features = eliminator.get_feature_names_out(predictors[:-1])

        self.saveToPickle(features, k, data_filename)