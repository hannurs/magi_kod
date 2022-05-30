import random
from matplotlib.pyplot import axis
from FeatureSelectionMethod import FeatureSelectionMethod
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.svm import SVC
import statsmodels.api as sm
import pandas as pd
from genetic_selection import GeneticSelectionCV
import numpy as np
import warnings
warnings.filterwarnings("ignore")

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
    def StepwiseSelectionMethod(self, Xdata, Ydata, k, threshold_in=0.01, threshold_out = 0.05, verbose=False):
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
                # print(worst_feature)
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
        features = self.StepwiseSelectionMethod(Xdata, Ydata, k)

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

class GeneticAlgorithm(FeatureSelectionMethod):
    @classmethod
    def saveKBestFeatures(self, k, data_filename):
        predictors, Xdata, Ydata = super().getTrainData(data_filename)

        svm_ = SVC(kernel="linear")
        selector = GeneticSelectionCV(svm_, cv=5, verbose=0,scoring="accuracy", max_features=k,n_population=100, crossover_proba=0.5,mutation_proba=0.2, n_generations=50,crossover_independent_proba=0.5,mutation_independent_proba=0.04,tournament_size=3, n_gen_no_change=10,caching=True, n_jobs=-1)
        selector.fit(Xdata, Ydata)

        features = selector.get_feature_names_out(predictors[:-1])
        self.saveToPickle(features, k, data_filename)

class SimulatedAnnealing(FeatureSelectionMethod):

    # @classmethod
    def modifySubset(features_subset, features):
        excluded = list(set(features.columns) - set(features_subset.columns))
        add_feature = random.choice([0,1])
        if add_feature and len(features_subset.columns) < len(features.columns):
            # print("dodaje")
            random_column = random.choice(excluded)
            features_subset = pd.concat([features_subset, features.loc[:, random_column]], axis=1)
        elif len(features_subset.columns) > 1:
            # print("usuwam")
            random_column = random.choice(list(set(features_subset.columns)))
            features_subset = features_subset.drop(random_column, axis=1)
        elif len(features_subset.columns) < len(features.columns):
            # print("dodaje")
            random_column = random.choice(excluded)
            features_subset = pd.concat([features_subset, features.loc[:, random_column]], axis=1)

        return features_subset


    @classmethod
    def SimulatedAnnealingMethod(self, k, data_filename, n_iterations=50, temperature=15):
        predictors, Xdata, Ydata = super().getTrainData(data_filename)
        Xdata_df = pd.DataFrame(Xdata, columns=predictors[:-1])
        features_subset_df = Xdata_df.sample(n=random.randint(int(len(predictors)/4), int(len(predictors)/3)), axis="columns")
        best_features_subset_df = features_subset_df
        svm_ = SVC(kernel="linear")
        # print(features_subset_df)
        # print()
        # print(features_subset_df.to_numpy())
        acc_best = svm_.fit(features_subset_df.to_numpy(), Ydata).score(features_subset_df.to_numpy(), Ydata)
        
        for it in range(n_iterations):
            # print(it)
            # # list(features_subset_df.columns)
            # print(list(features_subset_df.columns))
            # print(list(best_features_subset_df.columns))
            temp = temperature / float(it + 1)
            features_subset_df = self.modifySubset(features_subset_df, Xdata_df)
            knn = KNeighborsClassifier(n_neighbors=3)
            acc_it = svm_.fit(features_subset_df.to_numpy(), Ydata).score(features_subset_df.to_numpy(), Ydata)
            if acc_it > acc_best:
                acc_best = acc_it
                best_features_subset_df = features_subset_df
            else:
                prob = np.exp((acc_best - acc_it) / temp)
                if prob <= random.random():
                    acc_best = acc_it
                    best_features_subset_df = features_subset_df
        
        return list(best_features_subset_df.columns)

            

    @classmethod
    def saveKBestFeatures(self, k, data_filename):
        predictors, Xdata, Ydata = super().getTrainData(data_filename)
        features = self.SimulatedAnnealingMethod(k, data_filename)
        self.saveToPickle(features, k, data_filename)