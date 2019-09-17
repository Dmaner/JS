# -*-coding:utf-8-*-
# Project: CH06
# Filename: Logistic regression
# Author: DMAN
# Dataset: breast cancer
# reference: https://blog.csdn.net/XiaoXIANGZI222/article/details/55097570

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np


dataset = load_breast_cancer()
data, target = dataset["data"], dataset["target"]
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3)


class LogisticRegression:

    def __init__(self, data, target):
        self.maxiter = 5000
        # [w0, w1, w2,..., wn, b]
        data = np.concatenate((data, np.ones((len(data), 1))), axis=1)
        self.w = np.zeros((len(data[0]), 1))
        self.learning_rate = 0.001
        self.m = len(data)
        self.train(data, target)

    def logisitc(self, x):
        """
        prevent to overflow
        """
        r = np.maximum(x, 0)
        l = np.minimum(x, 0)
        l = np.exp(l) / (1 + np.exp(l))
        r = 1 / (1 + np.exp(-r))
        return l + r - 1 / 2

    def Loss(self, x, y):
        wx = np.dot(x, self.w)
        y_ = self.logisitc(wx)
        return y_ - np.mat(y).transpose()

    def train(self, data, target):
        """
        gradient descent
        """
        for i in range(self.maxiter):
            self.w -= self.learning_rate / self.m * np.dot(data.transpose(), self.Loss(data, target))

    def predict(self, x):
        x = np.append(x, 1).reshape((1, -1))
        wx = np.dot(x, self.w).flatten()[0]
        p = self.logisitc(wx)
        if p > 0.5:
            return 1
        else:
            return 0

    def test(self, data, target):
        total = len(data)
        correct = 0
        for x, y in zip(data, target):
            if self.predict(x) == y:
                correct += 1

        return correct / total


if __name__ == '__main__':
    model = LogisticRegression(x_train, y_train)
    print("Accuracy: {:.2f}%".format(100*model.test(x_test, y_test)))