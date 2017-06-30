def init():
    global globalMode
    globalMode=2
    global globalK
    globalK = 85
    global globalN
    globalN = 127
    global globalM
    globalM = globalN - globalK
    global globalSnr
    globalSnr = float(2)
    global globalInd
    globalInd = 2
    global globalNumOfWords
    globalNumOfWords= 100000
    global globalLogFolder
    globalLogFolder="/tmp/bch_model"
