import sklearn
from sklearn import datasets
import numpy
from sklearn.neural_network import MLPClassifier
import helper as h

digits = datasets.load_digits()
h.shuffle_in_unison(digits.data, digits.target)

train_set_x, cv_set_x, test_set_x = h.divide_groups(digits.data)
train_set_y, cv_set_y, test_set_y = h.divide_groups(digits.target)

clf = MLPClassifier(
        solver='lbfgs',
        alpha=1e-5,hidden_layer_sizes=(5, 2),
        random_state=1
        )

data = (train_set_x, train_set_y, cv_set_x, cv_set_y)
score = h.measure_accuracy(clf, data)

print("Score: " + str(score))

