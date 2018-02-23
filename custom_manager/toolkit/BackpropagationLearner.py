from BackpropNetwork import BackpropNetwork
from supervised_learner import SupervisedLearner
from matrix import Matrix
import math

class BackpropagationLearner(SupervisedLearner):
    learningRate = .1
    network = None
    bestNetworkSoFar = None
    currentAccuracy = None
    previousAccuaracy = None
    bestAccuaracy = None
    epochs = None
    epochsWithoutMeaningfulUpdate = None
    numberOfHiddenLayers = 1
    features = []
    labels = []
    accuracyDeltaCutoff = .01
    validationSetFeatures = []
    validationSetLabels = []
    trainingSetFeatures = []
    trainingSetLabels = []
    isTraining = False
    needResetBeforeTest = True

    layerSizesArray = [4, 8, 3]

    def __init__(self):
        self.network = BackpropNetwork(self.numberOfHiddenLayers + 2, self.learningRate, self.layerSizesArray)
        self.bestNetworkSoFar = None
        self.validationSetFeatures = []
        self.validationSetLabels = []
        self.trainingSetFeatures = []
        self.trainingSetLabels = []
        self.bestAccuaracy = 0
        self.currentAccuracy = 0
        self.previousAccuaracy = 0
        self.epochs = 0
        self.epochsWithoutMeaningfulUpdate = 0

    def checkAccuracyForMeaningfulUpdate(self):
        self.isTraining = False
        mses = []
        for i in range(len(self.validationSetFeatures)):
            errors = []
            targets = []
            for j in range(len(self.network.outputs)):
                if self.validationSetLabels[i] == j:
                    targets.append(1)
                else:
                    targets.append(0)
            self.predict(self.validationSetFeatures[i], targets)
            for j in range(len(self.network.outputs)):
                errors.append(targets[j] - self.network.outputs[j].output)
            sse = 0
            for error in errors:
                sse += error**2
            mse = sse/len(self.validationSetLabels)
            mses.append(mse)
        totalMse = 0
        for mse in mses:
            totalMse += mse
        totalMse = totalMse / len(mses)
        if totalMse < 1 - self.bestAccuaracy:
            self.bestNetworkSoFar = BackpropNetwork(self.numberOfHiddenLayers + 2, self.learningRate, self.layerSizesArray)
            moreLayers = False
            currLayer = self.bestNetworkSoFar.firstNodes
            copyMeLayer = self.network.firstNodes
            while moreLayers:
                for i in range(len(currLayer)):
                    for j in range(len(currLayer[i].forwardConnections)):
                        currLayer[i].forwardWeights[j] = copyMeLayer[i].forwardWeights[j]
                    if type(currLayer[0]) is type(currLayer.forwardConnections[0]):
                        currLayer = currLayer.forwardConnections[0]
                        copyMeLayer = copyMeLayer.forwardConnections[0]
                    else:
                        moreLayers = False
            self.bestAccuaracy = 1 - totalMse
            self.epochsWithoutMeaningfulUpdate = 0
        else:
            self.epochsWithoutMeaningfulUpdate += 1


    def train(self, features, labels):
        features.shuffle(labels)
        validationSetSize = int(features.rows * .25)
        for i in range(features.rows):
            if i > validationSetSize:
                self.validationSetFeatures.append(features.row(i))
                self.validationSetLabels.append(labels.row(i))
            else:
                self.trainingSetFeatures.append(features.row(i))
                self.trainingSetLabels.append(features.row(i))
        # while self.epochsWithoutMeaningfulUpdate < 5:
        while self.epochs < 10:
            features.shuffle(labels)
            for i in range(len(self.trainingSetFeatures)):
                input = self.trainingSetFeatures[i]
                correctAnswer = self.trainingSetLabels[i]
                self.isTraining = True
                targets = []
                for j in range(len(self.network.outputs)):
                    debugj = j
                    debuglabel = self.validationSetLabels[i][0]
                    if self.validationSetLabels[i][0] == j:
                        targets.append(1)
                    else:
                        targets.append(0)
                result = self.predict(input, targets)
            self.epochs = self.epochs + 1
            self.checkAccuracyForMeaningfulUpdate()
        self.isTraining = False
        print('Epochs: ' + str(self.epochs))
        print(self.bestAccuaracy)



    def predict(self, features, labels):
        targets = labels
        # if labels != []:
        #     targets = labels
        #labels = []
        self.network.resetNetwork
        self.network.processInput(features, targets, self.isTraining)
        output = 0
        outputIndex = 0
        for outputNode, i in zip(self.network.outputs, range(len(self.network.outputs))):
            if outputNode.output > output:
                outputIndex = i
                output = outputNode.output
        labels.append(outputIndex)
        return output