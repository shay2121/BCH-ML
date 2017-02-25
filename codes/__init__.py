import random as random
import numpy as np
import csv
from openpyxl import load_workbook
from array import array



def readCsvG():
  r=np.empty([globalK, globalN], dtype=int)
  wb=load_workbook('CFG/codes.xlsx')
  sheet = wb.get_sheet_by_name('bch'+str(globalK)+'x'+str(globalN))
  for i in range(2,2+globalK-1):
    for j in range(1, 1 + globalN-1):
      r[i-1,j]=sheet.cell(row=i,column=j).value.__int__()
  return r



def flip(p):
    return 0 if random.random() < 0.5 else 1

def generateCodeWord():
  R=np.true_divide(globalK,globalN)
  globalSnrUp=globalSnr+10*np.log10(2.0*R)
  factor = 2 * pow(10.0, (globalSnrUp / 20.0))
  noise=np.random.normal(0,1,globalN)/factor
  w= np.random.randint(2, size=globalK)
  x=np.remainder(np.dot(w,G),2)
  y=np.float16(x+noise)
  y=["%.4f" %xx for xx in y]
  answer=y+[x[globalInd]]
  row4Write=np.asarray(answer)
  return row4Write


def generateData(k,n,snr,numOfWords,ind,dataFileName1,dataFileName2):
  global globalInd
  globalInd=ind
  global globalK
  globalK=k
  global  globalN
  globalN= n
  global globalM
  globalM= globalN - globalK
  global globalSnr
  globalSnr= snr
  global globalNumOfWords
  globalNumOfWords= numOfWords
  global G
  step=globalNumOfWords / 100
  G = readCsvG();
  dataFile1 = open(dataFileName1, "wb")
  dataFile2 = open(dataFileName2, "wb")
  global dataWriter1
  global dataWriter2
  dataWriter1= csv.writer(dataFile1, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
  dataWriter2 = csv.writer(dataFile2, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE, )
  dataWriter1.writerow([globalNumOfWords-1,globalN+0,globalK,globalInd])
  dataWriter2.writerow([globalNumOfWords - 1, globalN + 0, globalK, globalInd])
  for i in range(1,globalNumOfWords):
        r=generateCodeWord()
        dataWriter1.writerow(r)
        r = generateCodeWord()
        dataWriter2.writerow(r)

  dataFile1.close()
  dataFile2.close()