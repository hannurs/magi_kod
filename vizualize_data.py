import os
import pickle
import matplotlib.pyplot as plt
from filter_feature_selection import read_file, FilterFSMethod

def vizualizeAndSave(method):
    dir_path_data = "databases/extracted/train"
    if method == FilterFSMethod.ReliefF:
        dir_path_pickle = "scores/filter/relief"
        ylabel_ = "ReliefF score"
        save_fig_dir = "figures/filter/relief/feature_importance_barchart/"
    elif method == FilterFSMethod.MutualInformation:
        dir_path_pickle = "scores/filter/mutual_information"
        ylabel_ = "Mutual Information score"
        save_fig_dir = "figures/filter/mutual_information/feature_importance_barchart/"
    elif method == FilterFSMethod.MaximalInformationCoefficient:
        dir_path_pickle = "scores/filter/maximal_information_coefficient"
        ylabel_ = "Maximal Information Coefficent score"
        save_fig_dir = "figures/filter/maximal_information_coefficient/feature_importance_barchart/"

    if len(os.listdir(dir_path_pickle)) == len(os.listdir(dir_path_data)):
        nr_files = len(os.listdir(dir_path_pickle))
    else:
        print("error")
        quit()

    for i in range(nr_files):
    # for pickle_file, data_file in os.listdir(dir_path_pickle), os.listdir(dir_path_data):
        predictors, Xdata, Ydata = read_file(dir_path_data + "/" + os.listdir(dir_path_data)[i])
        with open(dir_path_pickle + "/" + os.listdir(dir_path_pickle)[i], "rb") as f:
            feat_importances = pickle.load(f)
            feat_importances_sorted = sorted(feat_importances.items(), key=lambda x:x[1], reverse=True)
            print()
            print(os.listdir(dir_path_pickle)[i])
            n_pred = 0
            for feature in feat_importances_sorted:
                print(feature[0], feature[1])
                n_pred += 1
                if n_pred == 5:
                    break
            plt.figure(figsize=(18,10))
            plt.bar(predictors[:-1], list(feat_importances.values()))
            plt.xticks(rotation="vertical")
            plt.ylabel(ylabel_)
            plt.xlabel("Feature name")
            plt.savefig(save_fig_dir + os.listdir(dir_path_pickle)[i][:-4] + ".png")
        # plt.show()
#####

vizualizeAndSave(FilterFSMethod.MutualInformation)
vizualizeAndSave(FilterFSMethod.MaximalInformationCoefficient)
vizualizeAndSave(FilterFSMethod.ReliefF)