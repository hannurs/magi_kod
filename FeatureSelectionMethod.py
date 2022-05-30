from abc import ABC, abstractclassmethod, abstractmethod
import os
from sklearn import preprocessing
import csv
import numpy as np
import pickle
import matplotlib.pyplot as plt

class FeatureSelectionMethod(ABC):
    
    path_to_traindata_dir = "extracted/train"
    path_to_testdata_dir = "extracted/test"
    path_to_figures_dir = "figures"
    path_to_features = "selected_features"
    filterMethod = False

    @classmethod
    def saveToPickle(self, features, k, data_filename):
        with open(self.path_to_features + "/" + self.__name__ + "/" + data_filename[:-4] + str(k) + ".pkl", "wb") as f:
            pickle.dump(features, f)

    @classmethod
    def saveKBestFeatures(self, k, data_filename):
        print("blad")
        pass

    @classmethod
    def plotKBestFeatures(self, k, data_filename, show=False):
        if not self.filterMethod:
            return
        features = self.getKBestFeatures(k, data_filename)
        plt.figure(figsize=(19,10))
        # if type(features) is dict:
        plt.bar(features.keys(), list(features.values()))
        plt.xticks(rotation="vertical")
        plt.ylabel(self.__name__ + " score")
        plt.xlabel("Feature name")
        plt.savefig(self.path_to_figures_dir + "/" + self.__name__ + "/" + data_filename[:-4] + str(k) + ".png")

    @classmethod
    def getKBestFeatures(self, k, data_filename):
        if os.path.isfile(self.path_to_features + "/" + self.__name__ + "/" + data_filename[:-4] + str(k) + ".pkl") == False:
            # print("plik nie ị̣stnieje")
            # print(self.path_to_features + "/" + self.__name__ + "/" + data_filename[:-4] + str(k) + ".pkl")
            self.saveKBestFeatures(k, data_filename)
        with open(self.path_to_features + "/" + self.__name__ + "/" + data_filename[:-4] + str(k) + ".pkl", "rb") as f:
            return pickle.load(f)

    @classmethod
    def getData(self, path):
        with open(path, "r") as f:
            reader = csv.reader(f)
            predictors = list(reader)[0]
        
        alldata = np.loadtxt(open(path, "rb"), delimiter=",", skiprows=1)
        Xdata = alldata[:, :-1]
        Ydata = alldata[:, -1]

        return predictors, preprocessing.normalize(Xdata), Ydata

    @classmethod
    def getTrainData(self, data_filename):
        return self.getData(self.path_to_traindata_dir + "/" + data_filename)
    
    @classmethod
    def getTestData(self, data_filename):
        return self.getData(self.path_to_testdata_dir + "/" + data_filename)