from FilterMethods import ANOVA

print(ANOVA.getKBestFeatures(5, "csv_result-easy.csv"))
ANOVA.plotKBestFeatures(10, "csv_result-easy.csv")