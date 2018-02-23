from BackpropNetwork import BackpropNetwork as b

input = [0.3,0.7]
target = [0.1,1.0]

i = b(3, .1, [2, 2, 2, 2])

i.firstNodes[0].forwardWeights[0] = .2
i.firstNodes[0].forwardWeights[1] = .3

i.firstNodes[1].forwardWeights[0] = -.1
i.firstNodes[1].forwardWeights[1] = -.3

i.firstNodes[2].forwardWeights[0] = .1
i.firstNodes[2].forwardWeights[1] = -.2

firstHiddenLayer = i.firstNodes[0].forwardConnections

firstHiddenLayer[0].forwardWeights[0] = -.2
firstHiddenLayer[0].forwardWeights[1] = -.1

firstHiddenLayer[1].forwardWeights[0] = -.3
firstHiddenLayer[1].forwardWeights[1] = .3

firstHiddenLayer[2].forwardWeights[0] = .1
firstHiddenLayer[2].forwardWeights[1] = .2

secondHiddenLayer = firstHiddenLayer[0].forwardConnections

secondHiddenLayer[0].forwardWeights[0] = -.1
secondHiddenLayer[0].forwardWeights[1] = -.2

secondHiddenLayer[1].forwardWeights[0] = .3
secondHiddenLayer[1].forwardWeights[1] = -.3

secondHiddenLayer[2].forwardWeights[0] = .2
secondHiddenLayer[2].forwardWeights[1] = .1

def printWeights():
    print("%.10f"%i.firstNodes[2].forwardWeights[0])
    print("%.10f"%i.firstNodes[0].forwardWeights[0])
    print("%.10f"%i.firstNodes[1].forwardWeights[0])
    print("%.10f"%i.firstNodes[2].forwardWeights[1])
    print("%.10f"%i.firstNodes[0].forwardWeights[1])
    print("%.10f"%i.firstNodes[1].forwardWeights[1])
    print('\n')
    print("%.10f"%firstHiddenLayer[2].forwardWeights[0])
    print("%.10f"%firstHiddenLayer[0].forwardWeights[0])
    print("%.10f"%firstHiddenLayer[1].forwardWeights[0])
    print("%.10f"%firstHiddenLayer[2].forwardWeights[1])
    print("%.10f"%firstHiddenLayer[0].forwardWeights[1])
    print("%.10f"%firstHiddenLayer[1].forwardWeights[1])
    print('\n')
    print("%.10f"%secondHiddenLayer[2].forwardWeights[0])
    print("%.10f"%secondHiddenLayer[0].forwardWeights[0])
    print("%.10f"%secondHiddenLayer[1].forwardWeights[0])
    print("%.10f"%secondHiddenLayer[2].forwardWeights[1])
    print("%.10f"%secondHiddenLayer[0].forwardWeights[1])
    print("%.10f"%secondHiddenLayer[1].forwardWeights[1])

    print('\n')

def printErrors(i, a, b):
    print(a[0].delta)
    print(a[1].delta)
    print(b[0].delta)
    print(b[1].delta)
    print(i.outputs[0].delta)
    print(i.outputs[1].delta)


printWeights()
i.processInput(input, target, True)
print("outputs: ")
print(i.outputs[0].output)
print(i.outputs[1].output)
print("errors")
printErrors(i, firstHiddenLayer, secondHiddenLayer)
print("NEXT")
printWeights()
i.processInput(input, target, True)
print("outputs: ")
print(i.outputs[0].output)
print(i.outputs[1].output)
print("errors")
printErrors(i, firstHiddenLayer, secondHiddenLayer)
print("NEXT")
printWeights()
i.processInput(input, target, True)
# print("outputs: ")
# print(i.outputs[0].output)
# print(i.outputs[1].output)

j = 4