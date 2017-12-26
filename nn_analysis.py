import sklearn
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
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

alpha_exponents = [-10,-8,-6,-4,-2,0,2,4,6,8,10]
data = (train_set_x, train_set_y, cv_set_x, cv_set_y)
accuracy_scores = h.accuracy_scores_for_nn(data, alpha_exponents)

plt.bar(alpha_exponents, accuracy_scores)
plt.show()

best_index = numpy.argmax(accuracy_scores)
best_alpha = 10 ** alpha_exponents[best_index]
print("Best alpha: " + str(best_alpha))

clf = MLPClassifier(
        solver="lbfgs",
        alpha=best_alpha,
        random_state=1
        )

data = (train_set_x, train_set_y, test_set_x, test_set_y)
score = h.measure_accuracy(clf, data)
print("Final Accuracy Score Test Set:" + str(score))
