
#@return centered list
def centerData(list, mean):
    centeredData = []
    for item in list:
        centeredData.append(item - mean)
    return centeredData

#built just for our 2z2 case here, could expand it to be more
#@return list two lists of two to represent 2z2 matrix
def buildCovarianeMatrix(x, y, xMean, yMean):
    sum = 0
    n = 2
    for xFeature, yFeature in zip(x, y):
        sum = sum + ((xFeature - xMean)*yFeature - yMean)/(n - 1)


m = 5 #Number of instances in data set
n = 2 #Number of input features
p = 1 #Final number of principal components chosen

xValues = [.2, -1.1, 1, .5, -.6]
yValues = [-.3, 2, -2.2, -1, 1]
xMean = 0
yMean = -.1

#Step 1: center data around 0
xCentered = centerData(xValues, xMean)
yCentered = centerData(yValues, yMean)

#Step 2: calculate covariance matrix of centered data
covarianceMatrix = buildCovarianeMatrix(xCentered, yCentered, xMean, yMean)