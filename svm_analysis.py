import sklearn
from sklearn import datasets
from sklearn import svm
import numpy
import matplotlib.pyplot as plt
import helper as h

digits = datasets.load_digits()
h.print_digit_info(digits)

print("Random Shuffle:")
print("First digits: " + str(digits.target[:5]))
h.shuffle_in_unison(digits.data, digits.target)
print("First digits: " + str(digits.target[:5]))

train_set_x, cv_set_x, test_set_x = h.divide_groups(digits.data)
train_set_y, cv_set_y, test_set_y = h.divide_groups(digits.target)

print("training_set length: " + str(len(train_set_x)))
print("cv_set_x length: " + str(len(cv_set_x)))
print("test_set length: " + str(len(test_set_x)))

gamma_exponents = [-8,-7,-6,-5,-4,-3,-2,-1,0,1]
data = (train_set_x, train_set_y, cv_set_x, cv_set_y)
accuracy_scores = h.accuracy_scores_for_svm(data, gamma_exponents)

plt.bar(gamma_exponents, accuracy_scores)
plt.show()

best_index = numpy.argmax(accuracy_scores)
best_gamma = 10 ** gamma_exponents[best_index]
print("Best gamma: " + str(best_gamma))

clf = svm.SVC(gamma=best_gamma, C=100)

data = (train_set_x, train_set_y, test_set_x, test_set_y)
score = h.measure_accuracy(clf, data)
print("Final Accuracy Score Test Set:" + str(score))
