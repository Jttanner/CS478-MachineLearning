from matrix import Matrix
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy
import csv

class MLGraphing:

    def plotBinaryResults(self, features, labels, weights):
        xAxisRed = []
        yAxisRed = []
        xAxisBlue = []
        yAxisBlue = []
        for i in range(features.rows):
            if labels.data[i][0] == 0.0:
                xAxisRed.append(features.data[i][0])
                yAxisRed.append(features.data[i][1])
            else:
                xAxisBlue.append(features.data[i][0])
                yAxisBlue.append(features.data[i][0])

        truePlot = plt.plot(xAxisRed, yAxisRed, 'ro')
        plt.setp(truePlot, 'color', 'r')
        falsePlot = plt.plot(xAxisBlue, yAxisBlue, 'ro')
        plt.setp(falsePlot, 'color', 'b')
        falsePlot[0].color = 'b'
        plt.xlabel("Happiness")
        plt.ylabel("Cuteness")
        catLegend = patches.Patch(color='red', label='Cat')
        dogLegend = patches.Patch(color='blue', label='Dog')
        plt.legend(handles=[catLegend, dogLegend])
        self.drawLinearSeperationLine(self, weights)
        plt.show()


    def drawLinearSeperationLine(self, weights):
        #w1*x + w2*y = bias
        #y = bias/w2 - (w1/w2)*x
        w1 = weights[0]
        w2 = weights[1]
        bias = weights[2]
        x = numpy.arange(-2, 2)
        y = bias/w2 - (w1/w2)*x
        plt.plot(x,y)
        plt.axis([-2, 2, -2, 2])

    def calculateLinearRegressionLineInfo(self, features):
        info = []
        n = features.rows
        sum_xy = 0
        sum_x = 0
        sum_y = 0
        sum_xSquared = 0
        for i in range(0,features.rows):
            sum_xy += features.row(i)[0] * features.row(i)[1]
            sum_x += features.row(i)[0]
            sum_y += features.row(i)[1]
            sum_xSquared += features.row(i)[0] * features.row(i)[0]
            slope = (n*sum_xy - (sum_x * sum_y))/(n*sum_xSquared - (sum_x * sum_x))
        info.append(slope)
        offset = (sum_y - (slope * sum_x))/n
        info.append(offset)
        return info

    def drawLinearRegressionLine(self, features):
        lineInfo = self.calculateLinearRegressionLineInfo(self, features)
        m = lineInfo[0]
        x = numpy.arange(-1, 2)
        b = lineInfo[1]
        y = m * x + b
        min = Matrix.column_min(features, 0)
        max = Matrix.column_max(features, 1)
        plt.plot(x, y)

    def plotMisclassificationRate(self, accuracyAtEachEpoch, epochs):
        if len(accuracyAtEachEpoch) != epochs:
            print("Number of epochs and lengths of accuracy array must match")
            return
        x = numpy.arange(0, epochs)
        y = []
        for i in range(len(accuracyAtEachEpoch)):
            y.append(1 - accuracyAtEachEpoch[i])
        missClassPlot = plt.plot(x, y)
        plt.xlabel("Epoch")
        plt.ylabel("Average Misclassification Rate")
        plt.axis([0, epochs, 0, .2])
        plt.show()
        f = open('runInfo5.csv', 'w')
        with f:
            writer = csv.writer(f)
            writer.writerows([accuracyAtEachEpoch])

    def plotAverageMisclassificationRate(self):
        print('yay')

accuracyAtEachEpoch = []

with open('runinfo.csv', newline='') as f1:
    with open('runinfo.csv', newline='') as f2:
        with open('runinfo.csv', newline='') as f3:
            with open('runinfo.csv', newline='') as f4:
                with open('runinfo.csv', newline='') as f5:
                    reader1 = csv.reader(f1)
                    reader2 = csv.reader(f2)
                    reader3 = csv.reader(f3)
                    reader4 = csv.reader(f4)
                    reader5 = csv.reader(f5)
                    for row1 in reader1:
                        for row2 in reader2:
                            for row3 in reader3:
                                for row4 in reader4:
                                    for row5 in reader5:
                                        accuracyAtEachEpoch.append(sum(row1, row2, row3, row4, row5)/5)
epochs = len(accuracyAtEachEpoch)

