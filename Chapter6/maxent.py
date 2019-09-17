# -*-coding:utf-8-*-
# Project: CH06
# Filename: maxnet
# Author: DMAN
# Dataset: digits

from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from collections import defaultdict as ddict
import numpy as np


dataset = load_digits()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.1
)


class MaxEnt:
    def __repr__(self):
        return "Maximum entropy model"

    def __init__(self, data, target):
        self.classes = set(target)
        self.data = self.rebuild_features(data)
        self.target = target
        self.XY = ddict(int)  # record frequency of (X=x,Y=y)
        self.X = ddict(int)  # record frequency of (X=x)
        self.total = len(data)
        self.xy2id = ddict(int)
        self.e = 0.001
        self.n = None  # the number of (x, y)
        self.w = None
        self.lastw = None
        self.M = None
        self.sEf = None
        self.mEf = None

    def rebuild_features(self, features):

        new_features = []
        for feature in features:
            new_feature = []
            for i, f in enumerate(feature):
                new_feature.append(str(i) + '_' + str(f))
            new_features.append(new_feature)
        return new_features

    def load(self):
        """
        load data and consider f'value is the same with each sample
        """
        for x, y in zip(self.data, self.target):
            if len(x) != 0:
                for f in set(x):
                    self.XY[(f, y)] += 1
                    self.X[f] += 1
        self.n = len(self.XY)
        self.M = len(self.data[0])
        self.w = np.zeros(self.n)

    def cal_model_Ef(self):
        """
        calculate the characteristic expectation
        of model distribution
        """
        self.mEf = np.zeros(self.n)
        for sample in self.data:
            probs = self.cal_probality(sample)
            for prob, y in probs:
                for x in sample:
                    if (x, y) in self.xy2id:
                        id = self.xy2id[(x, y)]
                        self.mEf[id] += prob * self.X[x] / self.total

    def cal_sample_Ef(self):
        """
        calculate the characteristic expectation
        of sample distribution
        """
        self.sEf = ddict(float)
        for id, xy in enumerate(self.XY):
            self.sEf[id] = self.XY[xy] / self.total
            self.xy2id[xy] = id

    def cal_probality(self, x):
        """
        6.22 P85
        """
        probs = [self.pyx(y, x) for y in self.classes]
        Z = sum([prob for prob, _ in probs])
        return [(prob / Z, y) for prob, y in probs]

    def pyx(self, y, x):
        """
        calculate the probability of (y, x)
        """
        prob = 0
        for x_ in x:
            if (x_, y) in self.xy2id:
                id = self.xy2id[(x_, y)]
                prob += self.w[id] * self.XY[(x_, y)] / self.total

        return np.exp(prob), y

    def endcheck(self):
        for w1, w2 in zip(self.w, self.lastw):
            if w1 - w2 > self.e:
                return False
        return True

    def gis(self, iteration=1000):
        """
        Using generalized iterative scaling algorithm to train model
        """
        self.load()
        self.cal_sample_Ef()
        self.cal_model_Ef()
        self.lastw = self.w[:]
        for time in range(iteration):
            print("{}/ {}: training".format(time, iteration))
            for i in range(self.n):
                self.w[i] += 1 / self.M * np.log(self.sEf[i] / self.mEf[i])
            if not self.endcheck():
                break
            else:
                self.lastw = self.w[:]

    def predict(self, x):
        probs = self.cal_probality(x)
        return sorted(probs, reverse=True, key=lambda item: item[0])[0][1]

    def test(self, data, target):
        total = len(data)
        correct = 0
        data = self.rebuild_features(data)
        for x, y in zip(data, target):
            if y == self.predict(x):
                correct += 1

        return correct / total


if __name__ == "__main__":
    model = MaxEnt(x_train, y_train)
    model.gis()
    print("Accuracy: {:.2f}%".format(100 * model.test(x_test, y_test)))
