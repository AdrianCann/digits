import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))
import numpy
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

def print_digit_info(digits):
    print("Digit information")
    print("Digit data is  " + str(type(digits.data[1])))
    print("digits.data[1]")
    print(digits.data[1])
    print("Size of data[1]: " + str(digits.data[1].size))
    m = len(digits.data)
    print('m = ' + str(m) + ' digits total')

### Test shuffle ###

# https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
# Although apparently sklearn has a ShffleSplit as well
# from sklearn.model_selection import ShuffleSplit

def shuffle_in_unison(a, b):
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b)

### Divide into Training Set, Cross Validation, Test Set ###

def divide_groups(array, train_set_size=0.6, cv_size=0.2):
    m = len(array)
    first_index = int(m * train_set_size)
    second_index = int(m * (train_set_size + cv_size))
    train = array[:first_index]
    cv = array[first_index:second_index]
    test = array[second_index:]
    return (train, cv, test);

### Measuring Accuracy of Predictions ###

def measure_accuracy(clf, data):
    train_x, train_y, cv_x, cv_y = data
    clf.fit(train_x, train_y)
    cross_validation_predictions = clf.predict(cv_x)
    score = accuracy_score(cv_y, cross_validation_predictions)
    return score

### Select Gamma ###

def accuracy_scores_for_svm(data, gamma_exp, C=100):
    accuracy = []
    for exp in gamma_exp:
        gamma = 10 ** exp
        clf = svm.SVC(gamma=gamma, C=C)
        score = measure_accuracy(clf, data)
        print("Accuracy Score: " + str(score)) # percent correct
        accuracy.append(score)
    return accuracy

def accuracy_scores_for_nn(data, alpha_exp):
    accuracy = []
    for exp in alpha_exp:
        alpha = 10 ** exp
        clf = MLPClassifier(solver='lbfgs', alpha=alpha, random_state=1)
        score = measure_accuracy(clf, data)
        print("Accuracy Score: " + str(score)) # percent correct
        accuracy.append(score)
    return accuracy

def set_labels_nn(plt):
    plt.title("Accuracy for Alpha Values")
    plt.xlabel("Alpha Values (logarithmic)")
    plt.ylabel("Correct predictions (fraction)")

def set_labels_svm(plt):
    plt.title("Accuracy for Gamma Values")
    plt.xlabel("Gamma Values (logarithmic)")
    plt.ylabel("Correct predictions (fraction)")
