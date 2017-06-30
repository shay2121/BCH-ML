
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codes
import csv
import glob, os
import shutil
import NSA
import tensorflow as tf
import numpy as np
import settings
import pandas as pd

settings.init()
globalMode=settings.globalMode
globalK=settings.globalK
globalN=settings.globalN
globalM=settings.globalM
globalSnr=settings.globalSnr
globalInd=settings.globalInd
globalNumOfWords=settings.globalNumOfWords
globalLogFolder=settings.globalLogFolder
G=codes.Code(globalN,globalK)

BCH_TRAINING= "BCH" + str(globalK) + "x" + str(globalN) + "x" + str(globalInd) + "x" + str(globalNumOfWords) + "Training.csv"
BCH_TEST= "BCH" + str(globalK) + "x" + str(globalN) + "x" + str(globalInd) + "x" + str(globalNumOfWords) + "Test.csv"
BCH_CAL= "BCH" + str(globalK) + "x" + str(globalN) + "x" + str(globalInd) + "x" + str(globalNumOfWords) + "Cal.csv"



#emp1 = Employee("Zara", 2000)
 # codes.generateData(globalK, globalN, globalSnr, globalNumOfWords,globalInd,fname1,fname2)
 # fname='Data/bch' + str(globalK) + 'x' + str(globalN) + '.csv'
  #t = np.genfromtxt(fname, delimiter=',')
  #return t

def genData(fname1, fname2):
    G.generateData4Training(globalInd, globalSnr, globalNumOfWords, globalInd, fname1, fname2)


def main():
    folder='/'+ globalLogFolder
    for root, dirs, files in os.walk(folder):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
   # NSA.decode()
    #return 1
    # Load datasets.
    genData(BCH_TRAINING, BCH_TEST)
    #solver=NSA.Solver()
    #solver.decode()

    training_set = tf.contrib.learn.datasets.base.load_csv(filename=BCH_TRAINING, target_column=globalN,
                                                           target_dtype=np.int)
    test_set = tf.contrib.learn.datasets.base.load_csv(filename=BCH_TEST, target_dtype=np.int)

    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=globalN)]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[100,30,2],
                                                n_classes=2,
                                                model_dir=globalLogFolder)

    # Fit model.
    classifier.fit(x=training_set.data,
                   y=training_set.target,
                   steps=1000)

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(x=test_set.data,
                                         y=test_set.target)["accuracy"]
    print('Accuracy: {0:f}'.format(accuracy_score))

    print("the end")
    # Classify two new flower samples.
    #  new_samples = np.array(
    #     [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
    # y = classifier.predict(new_samples)
    # print('Predictions: {}'.format(str(y)))


    z = classifier.predict(test_set.data)
    z = np.delete(z, [globalNumOfWords-1], None)
    csv_input = pd.read_csv(BCH_TEST)
    A = np.vstack([np.transpose(csv_input.as_matrix()),z])
    A=np.transpose(A[[globalInd,globalN,globalN+1]])
    print("aa")
    np.savetxt(BCH_CAL, A, delimiter=",")

    
main()

if __name__ == "__main__": main()


