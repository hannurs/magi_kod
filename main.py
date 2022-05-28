from FilterMethods import ANOVA, MaximalInformationCoefficient, MutualInformation, ReliefF

print(ReliefF.getKBestFeatures(5, "csv_result-easy.csv"))
# MutualInformation.plotKBestFeatures(7, "csv_result-easy.csv")