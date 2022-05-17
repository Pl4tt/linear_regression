import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
from linear_regression import LinearRegression


class MultipleRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n = len(self.x)
    
    def get_line(self):
        x = np.append(np.ones((self.n, 1), dtype=float), self.x, axis=1)

        beta_first = np.linalg.inv(np.dot(x.transpose(), x))
        coefficients = np.dot(np.dot(beta_first, x.transpose()), self.y)

        return coefficients[1:], coefficients[0]
    
    def predict(self, x):
        coefficients, intercept = self.get_line()
        coefficients = np.append([intercept], coefficients)
        x = np.append([1], x)

        return np.dot(x, coefficients)


def visual_example():
    data = pd.read_csv("student-mat.csv", sep=";")
    data = data[["G1", "G2", "G3"]]

    predict = "G3"

    x = np.array(data.drop([predict], axis=1))
    y = np.array(data[predict])

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = MultipleRegression(x, y)
    coefficients, intercept = linear.get_line()
    print(coefficients, intercept)
    for i, test in enumerate(x_test):
        print(f"Error: {abs(linear.predict(test) - y_test[i])}\n\r", linear.predict(test), y_test[i])
    
    # 3d plot
    figure = plt.figure()
    graph_3d = figure.add_subplot(projection="3d")

    first = "G1"
    second = "G2"

    # labels
    graph_3d.scatter(data[first], data[second], data[predict])
    graph_3d.set_xlabel("First Grade")
    graph_3d.set_ylabel("Second Grade")
    graph_3d.set_zlabel("Final Grade")

    # best fit line
    x3, y1, z1 = 0, 0, intercept
    x2, y2, z2 = max(data[first]), max(data[second]), max(data[first])*coefficients[0] + max(data[second])*coefficients[1] + intercept
    graph_3d.plot([x3, x2], [y1, y2], zs=[z1, z2], color="red")

    x3, y3, z3 = max(data[first]), 0, max(data[first])*coefficients[0] + intercept
    x4, y4, z4 = 0, max(data[second]), max(data[second])*coefficients[1] + intercept
    graph_3d.plot([x3, x4], [y3, y4], zs=[z3, z4], color="red")

    plt.show()

def multivariable_example():
    data = pd.read_csv("student-mat.csv", sep=";")
    data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
    predict = "G3"

    x = np.array(data.drop([predict], axis=1))
    y = np.array(data[predict])

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = MultipleRegression(x, y)
    coefficients, intercept = linear.get_line()
    print(coefficients, intercept)

    for i, test in enumerate(x_test):
        print(f"Error: {abs(linear.predict(test) - y_test[i])}\n\r", linear.predict(test), y_test[i])


if __name__ == "__main__":
    multivariable_example()