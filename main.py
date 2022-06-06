from FilterMethods import ANOVA, MaximalInformationCoefficient, MutualInformation, ReliefF
from WrapperSelectionMethods import ForwardSelection, BackwardSelection, GeneticAlgorithm, RecursiveFeatureElimination, SimulatedAnnealing, StepwiseSelection
import os
import sklearn

path_to_files = "extracted/train"
print('The scikit-learn version is {}.'.format(sklearn.__version__))
for k in range(5, 15, 2):
    for filename in os.listdir(path_to_files):
        # print("ANOVA: ", k, " features, ", filename)
        # print(ANOVA.getKBestFeatures(k, filename))
        # print()
        # print("ReliefF: ", k, " features, ", filename)
        # print(ReliefF.getKBestFeatures(k, filename))
        # print()
        # print("MutualInformation: ", k, " features, ", filename)
        # print(MutualInformation.getKBestFeatures(k, filename))
        # print()
        # print("MaximalInformationCoefficient: ", k, " features, ", filename)
        # print(MaximalInformationCoefficient.getKBestFeatures(k, filename))
        # print()
        # print("ForwardSelection: ", k, " features, ", filename)
        # print(ForwardSelection.getKBestFeatures(k, filename))
        # print()
        # print("BackwardSelection: ", k, " features, ", filename)
        # print(BackwardSelection.getKBestFeatures(k, filename))
        # print()
        print("StepwiseSelection: ", k, " features, ", filename)
        print(StepwiseSelection.getKBestFeatures(k, filename))
        print()
        # print("RecursiveFeatureElimination: ", k, " features, ", filename)
        # print(RecursiveFeatureElimination.getKBestFeatures(k, filename))
        # print()
        # print("GeneticAlgorithm: ", k, " features, ", filename)
        # print(GeneticAlgorithm.getKBestFeatures(k, filename))
        # print()
        # print("SimulatedAnnealing: ", k, " features, ", filename)
        # print(SimulatedAnnealing.getKBestFeatures(k, filename))
        # print()