from FilterMethods import ANOVA, MaximalInformationCoefficient, MutualInformation, ReliefF
from WrapperSelectionMethods import ForwardSelection, BackwardSelection, GeneticAlgorithm, RecursiveFeatureElimination, StepwiseSelection

print(StepwiseSelection.getKBestFeatures(5, "csv_result-easy.csv"))
print(RecursiveFeatureElimination.getKBestFeatures(5, "csv_result-easy.csv"))
print(GeneticAlgorithm.getKBestFeatures(5, "csv_result-easy.csv"))
# print(BackwardSelection.getKBestFeatures(5, "csv_result-easy.csv"))
# MutualInformation.plotKBestFeatures(7, "csv_result-easy.csv")