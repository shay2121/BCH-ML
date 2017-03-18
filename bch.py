
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

mode =2
globalK=85
globalN= 127
globalM=globalN-globalK
globalSnr=float(40)
globalInd=3
numOfWords=10000
logFolder="/tmp/bch_model"
BCH_TRAINING= "BCH" + str(globalK) + "x" + str(globalN) + "Training.csv"
BCH_TEST= "BCH" + str(globalK) + "x" + str(globalN) + "test.csv"


def genData(fname1,fname2):
    c1=codes.Code(globalN,globalK)
    c1.generateData4Training(globalInd,globalSnr,numOfWords,globalInd,fname1,fname2)

#emp1 = Employee("Zara", 2000)
 # codes.generateData(globalK, globalN, globalSnr, numOfWords,globalInd,fname1,fname2)
 # fname='Data/bch' + str(globalK) + 'x' + str(globalN) + '.csv'
  #t = np.genfromtxt(fname, delimiter=',')
  #return t


def main():
    folder='/'+logFolder
    for root, dirs, files in os.walk(folder):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
   # NSA.decode()
    #return 1
    # Load datasets.
    genData(BCH_TRAINING, BCH_TEST)
    solver=NSA.Solver()
    solver.decode()

    training_set = tf.contrib.learn.datasets.base.load_csv(filename=BCH_TRAINING, target_column=globalN,
                                                           target_dtype=np.int)
    test_set = tf.contrib.learn.datasets.base.load_csv(filename=BCH_TEST, target_dtype=np.int)

    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=globalN)]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[4000, 80],
                                                n_classes=2,
                                                model_dir=logFolder)

    # Fit model.
    classifier.fit(x=training_set.data,
                   y=training_set.target,
                   steps=10)

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


    print("start train...")



if __name__ == "__main__": main()


