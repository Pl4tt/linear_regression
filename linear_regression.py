import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.mean_x = sum(x)/len(x)
        self.mean_y = sum(y)/len(y)
    
    def get_line(self):
        if len(self.x) != len(self.y):
            return

        Sxy = sum([x*y for x, y in zip(self.x, self.y)]) - len(self.x)*self.mean_x*self.mean_y
        Sxx = sum([x**2 for x in self.x]) - len(self.x)*self.mean_x**2

        coefficient = Sxy/Sxx
        intercept = self.mean_y - coefficient*self.mean_x

        return float(coefficient), float(intercept)
    
    def predict(self, x):
        coef, intercept = self.get_line()

        return coef*x + intercept



if __name__ == "__main__":
    data = pd.read_csv("student-mat.csv", sep=";")
    data = data[["G1", "G2"]]

    predict = "G2"

    x = np.array(data.drop([predict], axis=1))
    y = np.array(data[predict])

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = LinearRegression(x, y)

    for i, test in enumerate(x_test):
        print(linear.predict(test), y_test[i])

    plt.scatter(data["G1"], data[predict])
    plt.xlabel("G1")
    plt.ylabel(predict)

    coef, intercept = linear.get_line()
    x1, y1 = 0, intercept
    x2, y2 = max(data["G1"]), max(data["G1"])*coef + intercept
    plt.plot([x1, x2], [y1, y2], color="r")

    plt.show()