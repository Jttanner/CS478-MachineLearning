from matrix import Matrix
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy

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


