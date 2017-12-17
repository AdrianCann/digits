import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))
import numpy
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

from sklearn.metrics import accuracy_score

digits = datasets.load_digits()
print("Digit information")
print("Digit data is  " + str(type(digits.data[1])))
print("digits.data[1]")
print(digits.data[1])
print("Size of data[1]: " + str(digits.data[1].size))

m = len(digits.data)
print('m = ' + str(m) + ' digits total')

# https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
# Although apparently sklearn ahs a ShffleSplit as well
# from sklearn.model_selection import ShuffleSplit

def shuffle_in_unison(a, b):
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b)

print("First digits: " + str(digits.target[:5]))
shuffle_in_unison(digits.data, digits.target)
print("First digits: " + str(digits.target[:5]))

first_index = int(m * 0.6)
training_set_x = digits.data[0:first_index]
training_set_y = digits.target[0:first_index]
second_index = int(m * 0.8)
cross_validation_set_x = digits.data[first_index:second_index]
cross_validation_set_y = digits.target[first_index:second_index]
test_set = digits.data[second_index:]

print("training_set length: " + str(len(training_set_x)))
print("cross_validation_set_x length: " + str(len(cross_validation_set_x)))
print("test_set length: " + str(len(test_set)))

clf = svm.SVC(gamma=0.001, C=100)

clf.fit(training_set_x, training_set_y)
cross_validation_predictions = clf.predict(cross_validation_set_x)
score = accuracy_score(cross_validation_predictions, cross_validation_set_y)
print("Accuracy Score: " + str(score)) # percent correct
