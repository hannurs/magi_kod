from sklearn.neighbors import KNeighborsClassifier
import sklearn.feature_selection
from filter_feature_selection import read_file
from enum import Enum
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

class WrapperFSMethod(Enum):
    ForwardSelection = 0
    BackwardSelection = 1
    MaximalInformationCoefficient = 2

def selectFeatures(method, n_selected_features):
    dir_path = "databases/extracted/train"
    for filename in os.listdir(dir_path):
        predictors, Xdata, Ydata = read_file(dir_path + "/" + filename)
        features = []
        if method == WrapperFSMethod.ForwardSelection:
            pickle_path = "scores/wrapper/forward_selection/"
            pickle_name = "ForwardSelection"
            knn = KNeighborsClassifier(n_neighbors=3)
            sfs = sklearn.feature_selection.SequentialFeatureSelector(knn, n_features_to_select=n_selected_features)
            sfs.fit(Xdata, Ydata)
            for i in range(len(sfs.get_support())):
                if sfs.get_support()[i]:
                    features.append(predictors[i])
        elif method == WrapperFSMethod.BackwardSelection:
            pickle_path = "scores/wrapper/backward_selection/"
            pickle_name = "ForwardSelection"
            knn = KNeighborsClassifier(n_neighbors=3)
            # print("jestem tu")
            sfs = sklearn.feature_selection.SequentialFeatureSelector(knn, n_features_to_select=n_selected_features, direction="backward")
            # print("jestem tu2")
            sfs.fit(Xdata, Ydata)
            # print("jestem tu3")
            features = sfs.get_feature_names_out(predictors[:-1])
            # for i in range(len(sfs.get_support())):
            #     if sfs.get_support()[i]:
            #         features.append(predictors[i])
        
        

        print(filename)
        for i in range(len(features)):
            print(features[i])
        with open(pickle_path + filename[:-9] + pickle_name + str(n_selected_features) + ".pkl", "wb") as f:
            pickle.dump(features, f)
        
        
    
for n_features in range(5, 16, 3):
    print(n_features)
    selectFeatures(WrapperFSMethod.BackwardSelection, n_features)

