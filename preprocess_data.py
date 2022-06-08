import os
import csv
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split

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

def extractHowWeTypeData0():
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

def extractHowWeTypeData():
    path_to_motion_capture = "databases/How We Type/motion_capture/Motion Capture/Motion Capture"
    path_to_typing = "databases/How We Type/typing/Typing/Typing"
    all_user_data_mc = []
    i = 0
    for user_file in os.listdir(path_to_motion_capture):
        user_data_mc = pd.read_csv(path_to_motion_capture + "/" + user_file, skiprows=2, sep="\t", quoting=3, error_bad_lines=False)
        user_data_mc.dropna(subset=["input_index"], inplace=True)
        print(user_data_mc["uid"].iloc[0])
        if 5307 in set(user_data_mc["uid"]):
            all_user_data_mc = user_data_mc
        else:
            all_user_data_mc = pd.concat([all_user_data_mc, user_data_mc], ignore_index=True)

    extracted = all_user_data_mc[["stimulus_index", "input", "finger", "iki", "Hands_L_Cout_x", "Hands_L_Cout_y", "Hands_L_Cout_z", "Hands_R_Cout_x", "Hands_R_Cout_y", "Hands_R_Cout_z", "Hands_L_R2_x", "Hands_L_R2_y", "Hands_L_R2_z", "Hands_L_M1_x", "Hands_L_M1_y", "Hands_L_M1_z", "Hands_L_I1_x", "Hands_L_I1_y", "Hands_L_I1_z", "Hands_L_Cin_x", "Hands_L_Cin_y", "Hands_L_Cin_z", "Hands_L_R1_x", "Hands_L_R1_y", "Hands_L_R1_z", "Hands_L_L1_x", "Hands_L_L1_y", "Hands_L_L1_z", "Hands_L_L2_x", "Hands_L_L2_y", "Hands_L_L2_z", "Hands_L_Aout_x", "Hands_L_Aout_y", "Hands_L_Aout_z", "Hands_L_I2_x", "Hands_L_I2_y", "Hands_L_I2_z", "Hands_R_L2_x", "Hands_R_L2_y", "Hands_R_L2_z", "Hands_L_M2_x", "Hands_L_M2_y", "Hands_L_M2_z", "Hands_L_T2_x", "Hands_L_T2_y", "Hands_L_T2_z", "Hands_R_R3_x", "Hands_R_R3_y", "Hands_R_R3_z", "Hands_L_I3_x", "Hands_L_I3_y", "Hands_L_I3_z", "Hands_L_R4_x", "Hands_L_R4_y", "Hands_L_R4_z", "Hands_L_T1_x", "Hands_L_T1_y", "Hands_L_T1_z", "Hands_L_R3_x", "Hands_L_R3_y", "Hands_L_R3_z", "Hands_L_M4_x", "Hands_L_M4_y", "Hands_L_M4_z", "Hands_L_L4_x", "Hands_L_L4_y", "Hands_L_L4_z", "Hands_L_L3_x", "Hands_L_L3_y", "Hands_L_L3_z", "Hands_L_M3_x", "Hands_L_M3_y", "Hands_L_M3_z", "Hands_R_Win_x", "Hands_R_Win_y", "Hands_R_Win_z", "Hands_R_Cin_x", "Hands_R_Cin_y", "Hands_R_Cin_z", "Hands_R_L1_x", "Hands_R_L1_y", "Hands_R_L1_z", "Hands_L_Win_x", "Hands_L_Win_y", "Hands_L_Win_z", "Hands_L_Wout_x", "Hands_L_Wout_y", "Hands_L_Wout_z", "Hands_R_I1_x", "Hands_R_I1_y", "Hands_R_I1_z", "Hands_L_Ain_x", "Hands_L_Ain_y", "Hands_L_Ain_z", "Hands_R_I2_x", "Hands_R_I2_y", "Hands_R_I2_z", "Hands_R_L3_x", "Hands_R_L3_y", "Hands_R_L3_z", "Hands_R_R2_x", "Hands_R_R2_y", "Hands_R_R2_z", "Hands_R_R1_x", "Hands_R_R1_y", "Hands_R_R1_z", "Hands_R_M3_x", "Hands_R_M3_y", "Hands_R_M3_z", "Hands_R_Wout_x", "Hands_R_Wout_y", "Hands_R_Wout_z", "Hands_R_Ain_x", "Hands_R_Ain_y", "Hands_R_Ain_z", "Hands_R_T2_x", "Hands_R_T2_y", "Hands_R_T2_z", "Hands_R_T3_x", "Hands_R_T3_y", "Hands_R_T3_z", "Hands_R_T1_x", "Hands_R_T1_y", "Hands_R_T1_z", "Hands_R_M1_x", "Hands_R_M1_y", "Hands_R_M1_z", "Hands_R_M2_x", "Hands_R_M2_y", "Hands_R_M2_z", "Hands_R_Aout_x", "Hands_R_Aout_y", "Hands_R_Aout_z", "Hands_R_M4_x", "Hands_R_M4_y", "Hands_R_M4_z", "Hands_R_L4_x", "Hands_R_L4_y", "Hands_R_L4_z", "Hands_R_R4_x", "Hands_R_R4_y", "Hands_R_R4_z", "Hands_L_I4_x", "Hands_L_I4_y", "Hands_L_I4_z", "Hands_L_T4_x", "Hands_L_T4_y", "Hands_L_T4_z", "Hands_R_I3_x", "Hands_R_I3_y", "Hands_R_I3_z", "Hands_R_T4_x", "Hands_R_T4_y", "Hands_R_T4_z", "Hands_R_I4_x", "Hands_R_I4_y", "Hands_R_I4_z", "Hands_L_T3_x", "Hands_L_T3_y", "Hands_L_T3_z", "uid"]]
    extracted.to_csv("out.csv", sep="\t")


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

    alldata = pd.read_csv(filename, delimiter="\t")
    alldata = alldata[:, 2:]
    # alldata = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)

    print(alldata)
    train_data = []
    test_data = []

    for sample_ in alldata:
        print(sample_)
        if random.random() <= ratio:
            train_data.append(sample_)
        else:
            test_data.append(sample_)

def splitHowWeType():
    df = pd.read_csv("out.csv", sep="\t")
    df.drop('Unnamed: 0', inplace=True, axis=1)
    test_samples = random.sample(range(0, 50), 10)
    test = df.apply(lambda row: row[df["stimulus_index"].isin(test_samples)])
    # train = df.apply(lambda row: row[df["stimulus_index"].(test_samples)])
    train = pd.concat([df,test]).drop_duplicates(keep=False)

    test.drop("stimulus_index", inplace=True, axis=1)
    train.drop("stimulus_index", inplace=True, axis=1)

    test.to_csv("extracted/test/HowWetypeTEST.csv", sep="\t")
    train.to_csv("extracted/train/HowWetypeTRAIN.csv", sep="\t")

splitHowWeType()
# extractHowWeTypeData()
    # with open(filename[:-4] + "TRAIN.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(train_data)
    # with open(filename[:-4] + "TEST.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(test_data)
# extractHowWeTypeData()
# print(df.columns)
# df = df.iloc[: , 1:]
# print(df)

# for file in os.listdir("extracted/train"):
#     df = pd.read_csv("extracted/train/" + file, sep=",")
#     df.to_csv("extracted/train/" + file, sep="\t")
# for file in os.listdir("extracted/test"):
#     df = pd.read_csv("extracted/test/" + file, sep=",")
#     df.to_csv("extracted/test/" + file, sep="\t")