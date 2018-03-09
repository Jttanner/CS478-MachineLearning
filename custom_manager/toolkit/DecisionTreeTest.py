from DecisionTree import DecisionTree

features = [[1,0,0,3, 2], [2,1, 2,1,3], [3,2,3,2,2], [3, 2,0, 3, 1], [2,3,4,1,0]]
labels = [3, 1, 0, 2, 0]  #for now

tree = DecisionTree(features, labels)

test = []

for feature in features:
    test.append(tree.getDecision(feature))

i = 4