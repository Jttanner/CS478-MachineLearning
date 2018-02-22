from BackpropNetwork import BackpropNetwork as b

input = [1, 1, 0, 0, 1]

i = b(3, input, .1, [5, 4, 8, 3], [1, 1, 0, 1])

i.processInput(input)

j = 4