# -*-coding:utf-8-*-
# Project: CH06
# Filename: maxnet
# Author: DMAN
# Dataset: digits

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict
import numpy as np


dataset = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.1
)


class MaxEnt(object):
    def init_params(self, X, Y):
        self.X_ = X
        self.Y_ = set()

        self.cal_Pxy_Px(X, Y)

        self.N = len(X)  # 训练集大小
        self.n = len(self.Pxy)  # 书中(x,y)对数
        self.M = 10000.0  # 书91页那个M，但实际操作中并没有用那个值
        # 可认为是学习速率

        self.build_dict()
        self.cal_EPxy()

    def build_dict(self):
        self.id2xy = {}
        self.xy2id = {}

        for i, (x, y) in enumerate(self.Pxy):
            self.id2xy[i] = (x, y)
            self.xy2id[(x, y)] = i

    def cal_Pxy_Px(self, X, Y):
        self.Pxy = defaultdict(int)
        self.Px = defaultdict(int)

        for i in range(len(X)):
            x_, y = X[i], Y[i]
            self.Y_.add(y)

            for x in x_:
                self.Pxy[(x, y)] += 1
                self.Px[x] += 1

    def cal_EPxy(self):
        """
        """
        self.EPxy = defaultdict(float)
        for id in range(self.n):
            (x, y) = self.id2xy[id]
            self.EPxy[id] = float(self.Pxy[(x, y)]) / float(self.N)

    def cal_pyx(self, X, y):
        result = 0.0
        for x in X:
            if self.fxy(x, y):
                id = self.xy2id[(x, y)]
                result += self.w[id]
        return (np.exp(result), y)

    def cal_probality(self, X):
        """
        """
        Pyxs = [(self.cal_pyx(X, y)) for y in self.Y_]
        Z = sum([prob for prob, y in Pyxs])
        return [(prob / Z, y) for prob, y in Pyxs]

    def cal_EPx(self):
        """
        """
        self.EPx = [0.0 for i in range(self.n)]

        for i, X in enumerate(self.X_):
            Pyxs = self.cal_probality(X)

            for x in X:
                for Pyx, y in Pyxs:
                    if self.fxy(x, y):
                        id = self.xy2id[(x, y)]

                        self.EPx[id] += Pyx * (1.0 / self.N)

    def fxy(self, x, y):
        return (x, y) in self.xy2id

    def train(self, X, Y):
        self.init_params(X, Y)
        self.w = [0.0 for i in range(self.n)]

        max_iteration = 1000
        for times in range(max_iteration):
            print("iterater times %d" % times)
            sigmas = []
            self.cal_EPx()

            for i in range(self.n):
                sigma = 1 / self.M * np.log(self.EPxy[i] / self.EPx[i])
                sigmas.append(sigma)

            # if len(filter(lambda x: abs(x) >= 0.01, sigmas)) == 0:
            #     break

            self.w = [self.w[i] + sigmas[i] for i in range(self.n)]

    def predict(self, testset):
        results = []
        for test in testset:
            result = self.cal_probality(test)
            results.append(max(result, key=lambda x: x[0])[1])
        return results


def rebuild_features(features):
    """
    """
    new_features = []
    for feature in features:
        new_feature = []
        for i, f in enumerate(feature):
            new_feature.append(str(i) + "_" + str(f))
        new_features.append(new_feature)
    return new_features


if __name__ == "__main__":
    x_train = rebuild_features(x_train)
    x_test = rebuild_features(x_test)
    model = MaxEnt()
    model.train(x_train, y_train)
    predicts = model.predict(x_test)
    print("Accuracy: {:.2f}%".format(100 * accuracy_score(y_test, predicts)))
