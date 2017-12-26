import sklearn
from sklearn import datasets
import numpy
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import helper as h

digits = datasets.load_digits()
h.shuffle_in_unison(digits.data, digits.target)

train_set_x, cv_set_x, test_set_x = h.divide_groups(digits.data)
train_set_y, cv_set_y, test_set_y = h.divide_groups(digits.target)
data = (train_set_x, train_set_y, cv_set_x, cv_set_y)

alpha_exponents = [-10,-8,-6,-4,-2,0,2,4,6,8,10]
accuracy_scores = h.accuracy_scores_for_nn(data, alpha_exponents)

plt.bar(alpha_exponents, accuracy_scores)
plt.show()

best_index = numpy.argmax(accuracy_scores)
best_alpha = 10 ** alpha_exponents[best_index]
print("Best alpha: " + str(best_alpha))

clf = MLPClassifier(
        solver='lbfgs',
        alpha=best_alpha,
        random_state=1
        )

data = (train_set_x, train_set_y, test_set_x, test_set_y)
score = h.measure_accuracy(clf, data)

print("Final Accuracy Score Test Set:" + str(score))
