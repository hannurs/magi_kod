from msilib.schema import Feature
from FeatureSelectionMethod import FeatureSelectionMethod
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.svm import SVC
import statsmodels.api as sm
from test import stepwise_selection
import pandas as pd

class ForwardSelection(FeatureSelectionMethod):
    @classmethod
    def saveKBestFeatures(self, k, data_filename):
        predictors, Xdata, Ydata = super().getTrainData(data_filename)
        features = []
        svm_ = SVC(kernel="linear")
        knn = KNeighborsClassifier(n_neighbors=3)
        selector = SequentialFeatureSelector(knn, n_features_to_select=k)
        # selector = SequentialFeatureSelector(knn, n_features_to_select=k)
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

class StepwiseSelection(FeatureSelectionMethod):
    @classmethod
    def saveKBestFeaturesSS(self, Xdata, Ydata, k, threshold_in=0.01, threshold_out = 0.05, verbose=True):
        included = []
        while True:
            changed=False
            # forward step
            excluded = list(set(Xdata.columns)-set(included))
            new_pval = pd.Series(index=excluded)
            for new_column in excluded:
                model = sm.OLS(Ydata, sm.add_constant(pd.DataFrame(Xdata[included+[new_column]]))).fit()
                new_pval[new_column] = model.pvalues[new_column]
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = excluded[new_pval.argmin()]
                included.append(best_feature)
                changed=True
                if verbose:
                    print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

            # backward step
            model = sm.OLS(Ydata, sm.add_constant(pd.DataFrame(Xdata[included]))).fit()
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
            if len(included) == k:
                break

        return included

    @classmethod
    def saveKBestFeatures(self, k, data_filename):
        predictors, Xdata, Ydata = super().getTrainData(data_filename)
        Xdata = pd.DataFrame(Xdata, columns=predictors[:-1])
        features = self.saveKBestFeaturesSS(Xdata, Ydata, k)

        self.saveToPickle(features, k, data_filename)


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