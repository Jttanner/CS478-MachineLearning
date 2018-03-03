

class DecisionTree:

    maxNumberOfClassificationForAttribute = 10
    featureInfo = []
    featureCount = 0

    def __init__(self, featureCount):
        self.featureCount = featureCount
        for i in range(featureCount):
            fillFeatureInfo = []
            for j in range(self.maxNumberOfClassificationForAttribute):
                fillFeatureInfo.append(0)
            self.featureInfo.append(fillFeatureInfo)

    def processFeatures(self, features):
        for feature, i in zip(features ,range(self.featureCount)):
            self.featureInfo[i][feature] += 1 #feature.value?
