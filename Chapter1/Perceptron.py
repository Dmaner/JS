# -*-coding:utf-8-*-
# Project: CH02
# Filename: perceptron
# Author: ? <smirk dot cao at gmail dot com>

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

data = load_breast_cancer()
x = data.data
y = data.target
number_features = len(x[0])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


class Model:
    def __init__(self):
        self.learning_rate = 0.001
        self.w = np.ones((1, number_features), np.float)
        self.b = 1

    def train(self, data, target):
        """
        :param data: numpy.array
        :param target: numpy.array
        :return: None
        """
        for x, y in zip(data, target):
            if (np.dot(self.w, x.T) + self.b) * y < 0:
                self.w += self.learning_rate * y * x
                self.b += self.learning_rate * y

    def test(self, data, target):
        """
        accuracy for perceptron machine
        :param data: numpy.array
        :param target: numpy.array
        :return: accuracy
        """
        total = 0
        correct = 0
        for x, y in zip(data, target):
            total += 1
            if np.sign(np.dot(self.w, x.T) + self.b) == y:
                correct += 1
            else:
                continue
        return correct / total


if __name__ == "__main__":
    model = Model()
    model.train(x_train, y_train)
    print("Accuracy: {:.2f}".format(model.test(x_test, y_test)))
