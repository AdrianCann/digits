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

alpha_exponents = [-2,0,2,4,6,8,10] # all action between -2, 2
data = (train_set_x, train_set_y, cv_set_x, cv_set_y)
accuracy_scores = h.accuracy_scores_for_nn(data, alpha_exponents)

plt.bar(alpha_exponents, accuracy_scores)
plt.title("Accuracy for Alpha Values")
plt.xlabel("Alpha Values (logarithmic)")
plt.ylabel("Correct predictions (fraction)")
plt.show()

alpha_exponents = numpy.arange(-4.0, 4.0, 0.5)
accuracy_scores = h.accuracy_scores_for_nn(data, alpha_exponents)
plt.bar(alpha_exponents, accuracy_scores, 0.9)
plt.title("Accuracy for Alpha Values")
plt.xlabel("Alpha Values (logarithmic)")
plt.ylabel("Correct predictions (fraction)")
plt.axis([-5.0,4.5,0.9,1.0,])
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
