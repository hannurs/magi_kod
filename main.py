from FilterMethods import ANOVA, MaximalInformationCoefficient, MutualInformation, ReliefF
from WrapperSelectionMethods import ForwardSelection, BackwardSelection

print(ForwardSelection.getKBestFeatures(5, "csv_result-easy.csv"))
print(BackwardSelection.getKBestFeatures(5, "csv_result-easy.csv"))
# MutualInformation.plotKBestFeatures(7, "csv_result-easy.csv")