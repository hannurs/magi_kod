import pickle
import numpy as np
import os
import csv

# f = open("scores/filter/reliefcsv_result-easyReliefF.pkl", "rb")
# output = pickle.load(f)
# print(output)
def read_file(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        predictors = list(reader)[0]
    
    alldata = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
    Xdata = alldata[:, :-1]
    Ydata = alldata[:, -1]

    return predictors, alldata


# i = 0
# for filename in os.listdir("databases/extracted/test"):
    
#     predictors, alldata = read_file("databases/extracted/test/" + filename)

#     if predictors[0] == "id":
#         alldata = np.delete(alldata, 0, 1)
#         predictors = predictors[1:]

#         with open("databases/extracted/test/" + filename, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(predictors)
#             writer.writerows(alldata)
#     i+=1

# i = 0
# for filename in os.listdir("databases/extracted/train"):
    
#     predictors, alldata = read_file("databases/extracted/train/" + filename)

#     if predictors[0] == "id":
#         alldata = np.delete(alldata, 0, 1)
#         predictors = predictors[1:]

#         with open("databases/extracted/train/" + filename, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(predictors)
#             writer.writerows(alldata)
#     i+=1

for i in range(4):
    f = open("scores/wrapper/forward_selection/" + os.listdir("scores/wrapper/forward_selection")[i], "rb")
    output = pickle.load(f)
    print(output)
    print()