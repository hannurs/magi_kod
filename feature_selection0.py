import numpy as np
import csv
import sklearn_relief

def read_file(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        predictors = list(reader)[0]
    
    alldata = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
    Xdata = alldata[:, :-1]
    Ydata = alldata[:, -1]

    return predictors, Xdata, Ydata

def ROCAUCpredictors(n_sel_pred, predictors, Xdata, Ydata):
    print()



if __name__ == "__main__":
    predictors, Xdata, Ydata = read_file("databases/extracted/GREYC_passphrasesTRAIN.csv")
    print(Xdata.shape)
    print(Ydata.shape)
    r = sklearn_relief.ReliefF()
    m = r.fit_transform(Xdata, Ydata)
    print(m)
