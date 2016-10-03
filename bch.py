import random as random
import numpy as np
import csv


globalK=3
globalN=7
globalM=globalN-globalK
snr=3
numOfWords=100000


dataFileName='Data/bch '+str(globalK)+'x'+str(globalN)+'.csv'
dataFile = open(dataFileName, "wb")
dataWriter = csv.writer(dataFile, delimiter='	', quotechar='"', quoting=csv.QUOTE_ALL)




def readCsvG():
  fname='CFG/codes.csv'
  t = np.genfromtxt(fname, delimiter=',')
  r=t[1:]
  return r




G=readCsvG();

def flip(p):
    return 0 if random.random() < 0.5 else 1

def generateCodeWord():
  R=np.true_divide(globalK,globalN)
  snrUp=snr+10*np.log10(2.0*R)
  factor = 2 * pow(10.0, (snrUp / 20.0))
  noise=np.random.normal(0,1,globalN)/factor
  w= np.random.randint(2, size=globalK)
  y=np.remainder(np.dot(w,G),2)
  x=y+noise
  dataWriter.writerow(y)
  dataWriter.writerow(x)
  return x


def main():
    print("- In the forest:")
    for i in range(1,numOfWords):
        generateCodeWord()
        if np.remainder(i,1000)==0:
            print(str(i/1000)+"%")
    print("end")


if __name__ == "__main__": main()