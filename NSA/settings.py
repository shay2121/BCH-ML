def init():
    global globalMode
    globalMode = 2
    global globalK
    globalK = 85
    global globalN
    globalN = 127
    global globalM
    globalM = globalN - globalK
    global globalSnr
    globalSnr = float(40)
    global globalInd
    globalInd = 3
    global numOfWords
    numOfWords = 100
    global logFolder
    logFolder = "/tmp/bch_model"
