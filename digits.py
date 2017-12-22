import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))
import numpy
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

from sklearn.metrics import accuracy_score

digits = datasets.load_digits()

def print_digit_info(digits):
    print("Digit information")
    print("Digit data is  " + str(type(digits.data[1])))
    print("digits.data[1]")
    print(digits.data[1])
    print("Size of data[1]: " + str(digits.data[1].size))
    m = len(digits.data)
    print('m = ' + str(m) + ' digits total')

print_digit_info(digits)

### Test shuffle ###

# https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
# Although apparently sklearn has a ShffleSplit as well
# from sklearn.model_selection import ShuffleSplit

def shuffle_in_unison(a, b):
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b)

print("First digits: " + str(digits.target[:5]))
shuffle_in_unison(digits.data, digits.target)
print("First digits: " + str(digits.target[:5]))

### Divide into Training Set, Cross Validation, Test Set ###

def divide_groups(array, train_set_size=0.6, cv_size=0.2):
    m = len(array)
    first_index = int(m * train_set_size)
    second_index = int(m * (train_set_size + cv_size))
    train = array[:first_index]
    cv = array[first_index:second_index]
    test = array[second_index:]
    return (train, cv, test);

train_set_x, cv_set_x, test_set_x = divide_groups(digits.data)
train_set_y, cv_set_y, test_set_y = divide_groups(digits.target)

print("training_set length: " + str(len(train_set_x)))
print("cv_set_x length: " + str(len(cross_validation_set_x)))
print("test_set length: " + str(len(test_set_x)))

### Measuring Accuracy of Predictions ###

def measure_accuracy(clf, train_x, train_y, cross_x, cross_y):
    clf.fit(train_x, train_y)
    cross_validation_predictions = clf.predict(cross_x)
    score = accuracy_score(cross_y, cross_validation_predictions)
    return score

### Select Gamma ###

gamma_exponents = [-8,-7,-6,-5,-4,-3,-2,-1,0,1]

def accuracy_scores_for(gamma_exp, C=100):
    accuracy = []
    for exp in gamma_exp:
        gamma = 10 ** exp
        clf = svm.SVC(gamma=gamma, C=C)
        score = measure_accuracy(clf, train_set_x, train_set_y, cv_set_x, cv_set_y)
        print("Accuracy Score: " + str(score)) # percent correct
        accuracy.append(score)
    return accuracy

accuracy_scores = accuracy_scores_for(gamma_exponents)

plt.bar(gamma_exponents, accuracy_scores)
plt.show()
# gamma best between 10 ** -6 and 10 ** -3
