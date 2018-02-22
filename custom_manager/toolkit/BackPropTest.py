from BackpropNetwork import BackpropNetwork as b

input = [1, 1, 0, 0, 1]
target = [1, 1, 0]

i = b(3, .1, [5, 4, 8, 3])

i.processInput(input, target, True)
i.processInput(input, target, True)
i.processInput(input, target, True)
i.processInput(input, target, True)
j = 4