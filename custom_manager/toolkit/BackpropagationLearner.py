from BackpropNetwork import BackpropNetwork
from supervised_learner import SupervisedLearner
from matrix import Matrix
import math
import csv

class BackpropagationLearner(SupervisedLearner):
    learningRate = .1
    network = None
    bestNetworkSoFar = None
    currentAccuracy = None
    previousAccuaracy = None
    bestAccuaracy = None
    epochs = None
    epochsWithoutMeaningfulUpdate = None
    numberOfHiddenLayers = 6
    features = []
    labels = []
    accuracyDeltaCutoff = .01
    validationSetFeatures = []
    validationSetLabels = []
    trainingSetFeatures = []
    trainingSetLabels = []
    isTraining = False
    needResetBeforeTest = True
    validationMSEs = []
    trainingMSEs = []
    testMSEs = []
    classificationAccuracies =[]
    lastMSE = 0


    #layerSizesArray = [4, 8, 3]
    layerSizesArray = [3, 6, 12, 18, 18, 12, 6, 2]

    def writeCSVFile(self, info, fileName):
        with open(str(fileName), 'w') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(info)

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
        totalMse = self.calculateValidationMSEs()
        self.validationMSEs.append(totalMse)
        #if totalMse < 1 - self.bestAccuaracy :
        test = totalMse - self.lastMSE
        test2 =  1 -  self.bestAccuaracy
        #if math.fabs(totalMse - self.lastMSE) > .000025:
        if 1 - totalMse > self.bestAccuaracy:
            #self.bestNetworkSoFar = BackpropNetwork(self.numberOfHiddenLayers + 2, self.learningRate, self.layerSizesArray)
            # moreLayers = False
            # currLayer = self.bestNetworkSoFar.firstNodes
            # copyMeLayer = self.network.firstNodes
            # while moreLayers:
            #     for i in range(len(currLayer)):
            #         for j in range(len(currLayer[i].forwardConnections)):
            #             currLayer[i].forwardWeights[j] = copyMeLayer[i].forwardWeights[j]
            #         if type(currLayer[0]) is type(currLayer.forwardConnections[0]):
            #             currLayer = currLayer.forwardConnections[0]
            #             copyMeLayer = copyMeLayer.forwardConnections[0]
            #         else:
            #             moreLayers = False
            self.bestAccuaracy = 1 - totalMse
            self.epochsWithoutMeaningfulUpdate = 0
        else:
            self.epochsWithoutMeaningfulUpdate += 1
        self.lastMSE = totalMse

    def calculateValidationMSEs(self):
        mses = []
        testPred = []
        testLabels = []
        for i in range(len(self.validationSetFeatures)):
            errors = []
            targets = []
            for j in range(len(self.network.outputs)):
                if self.validationSetLabels[i][0] == j:
                    targets.append(1)
                    testLabels.append(j)
                else:
                    targets.append(0)
            self.predict(self.validationSetFeatures[i], [])
            output = 0
            outputIndex = 0
            for j in range(len(self.network.outputs)):
                if self.network.outputs[j].output > output:
                    output = self.network.outputs[j].output
                    outputIndex = j
                errors.append(targets[j] - self.network.outputs[j].output)
            testPred.append(outputIndex)
            sse = 0
            for error in errors:
                sse += error**2
            mse = sse/len(errors)
            mses.append(mse)
        totalMse = 0
        for mse in mses:
            totalMse += mse
        totalMse = totalMse / len(mses)
        return totalMse

    def train(self, features, labels):
        self.isTraining = True
        # if features.rows < 4:
        #     self.predictForTraining(features.row(0), labels.row(0))
        #     return
        #features.shuffle(labels)
        #Matrix(data, 0, 0, train_size, data.cols-1)
        #features = Matrix(features, 0, 0, features.rows, features.cols -5)
        validationSetSize = int(features.rows * .25)
        for i in range(features.rows):
            if i > validationSetSize:
                self.validationSetFeatures.append(features.row(i))
                self.validationSetLabels.append(labels.row(i))
            else:
                self.trainingSetFeatures.append(features.row(i))
                self.trainingSetLabels.append(features.row(i))
        while self.epochsWithoutMeaningfulUpdate < 25 and self.epochs < 500:
        #while self.epochs < 300:
            print("current epoch: " + str(self.epochs))
            correct = 0
            total = 0
            features.shuffle(labels)
            self.features = features
            self.labels = labels
            mses = []
            for i in range(self.features.rows):
                sse = 0
                input = self.features.row(i)
                correctAnswer = self.labels.row(i)
                targets = []
                for k in range(11):
                    targets.append(1 if self.labels.row(i)[0] == k else 0)
                #targets = [1,0,0] if correctAnswer[0] == 0 else [0,1,0] if correctAnswer[0] == 1 else [0,0,1]
                self.predictForTraining(input, targets)
                for j in range(len(self.network.outputs)):
                    sse += (targets[j] - self.network.outputs[j].output)**2
                mses.append(sse/len(self.network.outputs))
                total += 1
                answerIndex = 0
                checkAnswer = 0
                for i in range(len(self.network.outputs)):
                    if checkAnswer < self.network.outputs[i].output:
                        checkAnswer = self.network.outputs[i].output
                        answerIndex = i
                correct += 1 if answerIndex == correctAnswer[0] else 0
            self.epochs = self.epochs + 1
            totalMSE = 0
            for mse in mses:
                totalMSE += mse
            totalMSE = totalMSE/len(mses)
            self.classificationAccuracies.append(correct/features.rows)
            self.trainingMSEs.append(totalMSE)
            self.checkAccuracyForMeaningfulUpdate()
        print('Epochs: ' + str(self.epochs))
        self.isTraining = False
        self.writeCSVFile(self.trainingMSEs, 'clicks_trainingMSEsnodes2.csv')
        self.writeCSVFile(self.validationMSEs,'clicks_validationMSEsnodes2.csv')
        self.writeCSVFile(self.classificationAccuracies, 'clicks_classificationAccuraciesnodes2.csv')


    def predictForTraining(self, features, targets):
        return self.network.processInput(features, targets, True)

    def predict(self, features, labels):
        self.network.processInput(features, [], False)
        output = 0
        outputIndex = 0
        for outputNode, i in zip(self.network.outputs, range(len(self.network.outputs))):
            if outputNode.output > output:
                outputIndex = i
                output = outputNode.output
        labels.append(outputIndex)
        return outputIndex