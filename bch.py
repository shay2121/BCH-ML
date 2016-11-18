
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codes
import csv


import tensorflow as tf
import numpy as np

globalK=3
globalN=7
globalM=globalN-globalK
globalSnr=3
globalInd=2
numOfWords=100000


def genData():
  codes.generateData(globalK, globalN, globalSnr, numOfWords,globalInd)
 # fname='Data/bch' + str(globalK) + 'x' + str(globalN) + '.csv'
  #t = np.genfromtxt(fname, delimiter=',')
  #return t


def main():
#    codes.generateData(globalK,globalN,globalSnr,numOfWords)
#    r=genData()
    print("start train...")


if __name__ == "__main__": main()


# Data sets
BCH_TRAINING= 'BCH3x7train.csv'
BCH_TEST = 'BCH3x7test.csv'

# Load datasets.

training_set = tf.contrib.learn.datasets.base.load_csv(filename=BCH_TRAINING, target_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv(filename=BCH_TEST, target_dtype=np.int)



# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=7)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/bch_model")

# Fit model.
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

print("game on")
# Classify two new flower samples.
#  new_samples = np.array(
#     [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
#y = classifier.predict(new_samples)
#print('Predictions: {}'.format(str(y)))
