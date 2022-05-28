from importlib.resources import path
import os
import csv
import numpy as np
import random
from enum import Enum
import pandas as pd

class dataset():
    X = [[]]
    Y = []
    predictors = []

def txtToCsv():
    pathToDir = "databases/How We Type/typing/Typing/Typing"
    for txtfilename in os.listdir(pathToDir):
        with open(pathToDir + "/" + txtfilename, "r") as f:
            lines = f.readlines()
        lines = [line.replace("\t", ",") for line in lines]
        with open("databases/How We Type/typing/Typing/comas/" + txtfilename, "w") as f:
            f.writelines(lines)
            # break

    pathToDir = "databases/How We Type/typing/Typing/comas"
    for txtfilename in os.listdir(pathToDir):
        file = pd.read_csv(pathToDir + "/" + txtfilename, on_bad_lines="skip")
        file.to_csv("databases/How We Type/typing/Typing/csv/" + txtfilename[:-3] + "csv", index=None)
    
    pathToDir = "databases/How We Type/motion_capture/Motion Capture/Motion Capture"
    for csvfilename in os.listdir(pathToDir):
        with open(pathToDir + "/" + csvfilename, "r") as f:
            lines = f.readlines()
        lines = [line.replace("\t", ",") for line in lines]
        with open("databases/How We Type/motion_capture/Motion Capture/comas/" + csvfilename, "w") as f:
            f.writelines(lines)
            # break

def extractHowWeTypeData():
    pathToDirTyping = "databases/How We Type/typing/Typing/csv"
    pathToDirMotionCapture = "databases/How We Type/motion_capture/Motion Capture/comas/"
    all_data = []
    for typing_file in os.listdir(pathToDirTyping):
        with open(pathToDirTyping + "/" + typing_file, "r") as f:
            lines_unsorted = f.readlines()[1:]
            # print(lines_unsorted[10])
            # lines_unsorted.sort(key=lambda x: int(x[2]))
            # print(lines_unsorted)
            # lines = sorted(lines_unsorted)
            lines = sorted(lines_unsorted, key=lambda x:x.split(",")[13])
        with open("databases/How We Type/typing/Typing/sorted/" + typing_file, "w", newline="") as f:
            for line in lines:
                f.write(line)
                # f.write("\n")

        # stimulus_id = 0
        # for line in lines:
        #     features = []
        #     param_values = line.split(",")
        #     user_id = int(param_values[2])

        #     if int(param_values[3]) == stimulus_id:
        #         wpm = param_values[9]
        #         sd_iki = param_values[10]
        #         stimulus_id = int(param_values[3])
                # features.append(wpm)
                # features.append(sd_iki)
                # features.append(user_id)
                # all_data.append(features)

    # with open("out.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(all_data)


def loadGREYCData():
    passphrasesDir = "databases\GREYC-Web based KeyStroke dynamics dataset\passphrases"
    sample_id = 0
    all_data = [[]]
    predictors = ["l_pp", "l_pr", "l_rp", "l_rr", "l_total"]
    # predictors = ["l_pp", "l_pr", "l_rp", "l_rr", "l_total", "p_pp", "p_pr", "p_rp", "p_rr", "p_total"]
    # all_data.append(predictors)
    features = []
    for userDir in os.listdir(passphrasesDir):
        print(userDir)
        for dateDir in os.listdir(passphrasesDir + "\\" + userDir):
            if os.path.isdir(passphrasesDir + "\\" + userDir + "\\" + dateDir):
                user_id = int(userDir[-3:])
                sample_data = []
                for filename in os.listdir(passphrasesDir + "\\" + userDir + "\\" + dateDir):
                    if filename[:-4] in predictors:
                        f = open(passphrasesDir + "\\" + userDir + "\\" + dateDir + "\\" + filename, "r")
                        lines = f.readlines()
                        no_feature = 0
                        for line in lines:
                            sample_data.append(line.strip())
                            if sample_id == 0:
                                features.append(filename[:-4] + str(no_feature))
                                no_feature += 1
                if sample_id == 0:
                    features.append("user_id")
                    all_data.append(features)
                    sample_id += 1
                sample_data.append(user_id)
                if len(sample_data) == len(features):
                    all_data.append(sample_data)
    all_data = all_data[1:]
    all_datamat = np.array(all_data)

    with open("GREYC_passphrases.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(all_data)
    return all_datamat

def splitDataset(filename, ratio):

    alldata = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)


    train_data = [[]]
    test_data = [[]]

    for sample_ in alldata:
        if random.random() <= ratio:
            train_data.append(sample_)
        else:
            test_data.append(sample_)

    with open(filename[:-4] + "TRAIN.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(train_data)
    with open(filename[:-4] + "TEST.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(test_data)

# splitDataset("databases/extracted/Keystroke Dynamics – Android platform.csv", 0.8)
# loadGREYCData()

# txtToCsv()
extractHowWeTypeData()