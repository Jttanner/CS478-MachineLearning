import sys
import datetime

readFile = open('humanClickData.txt', 'r')
writeFile = open("clicksJustTime.arff", 'a')

firstLine = True
lastTime = 0
for line in readFile:
    entry = ''
    date = ''
    commaCount = 0
    for char in line:
        if commaCount == 2 and char != '\n':
            date += char
            continue
        if char == '\n':
            if firstLine:
                lastTime = datetime.datetime.strptime(date, ' %Y-%m-%d %H:%M:%S.%f')
                firstLine = False
                continue
            formattedDate = datetime.datetime.strptime(date, ' %Y-%m-%d %H:%M:%S.%f')
            tempTime = formattedDate
            formattedDate = formattedDate - lastTime
            lastTime = tempTime
            entry = str(formattedDate.seconds) + '.' + str(formattedDate.microseconds)
            entry += ',human\n'
            writeFile.write(entry)
        if char == ' ':
            continue
        entry += char
        if char == ',':
            commaCount += 1

