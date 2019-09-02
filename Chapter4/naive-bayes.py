# -*-coding:utf-8-*-
# Project: CH03
# Filename: perceptron
# Author: DMAN
# Dataset: breast cancer

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

data = load_breast_cancer()
x = data["data"]
y = data["target"]
number_features = len(x[0])

# from pprint import pprint
# import matplotlib.pyplot as plt

# def show(x):
#     image = x.reshape(8, 8)
#     plt.imshow(image, cmap="gray")
#     plt.axis("off")
#     plt.show()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


class Model:
    """Naive bayes by using gaussian distribution"""

    def __init__(self, data, target):
        self.classes = set(target)
        self
        # preprocess data
        data = self.preprocess(data)

        # Priori probability
        # just 1/10 for each digit

        # Conditional probability
        self.dict = {}
        for i in range(10):
            self.dict[i] = sum(target == i)/len(target)

        self.lambdas = {}
        for i in range(10):
            k = data[target == i]
            for j in range(number_features):
                # get lambdas
                self.lambdas[(j, i)] = sum(k[:,j])/len(k)

    def predict(self, x):
        x[x > 0] = 1
        list = []
        for i in range(10):
            p = self.dict[i]
            for j in range(number_features):
                p  = p*(x[j]*self.lambdas[(j, i)]+(1-x[j])*(1-self.lambdas[(j, i)]))
            list.append((p, i))
        sorted(list, reverse=True, key=lambda x: x[0])
        return list[0][1]

    def preprocess(self, data):
        for i in range(len(data)):
            x = data[i]
            x[x > 0] = 1
            data[i] = x
        return data

    def test(self, testdata, testtarget):
        total = 0
        correct = 0
        for x, y in zip(testdata, testtarget):
            if self.predict(x) == y:
                correct += 1
            total += 1
        return correct / total

if __name__ == "__main__":
    # model = naive_bayes.BernoulliNB()
    # model.fit(x_train,y_train)
    # print("Accuracy: {:.2f}%".format(model.score(x_test, y_test)*100))
    model = Model(x_train, y_train)
    print("Accuracy: {:.2f}%".format(100*model.test(x_test, y_test)))