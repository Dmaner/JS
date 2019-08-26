# -*-coding:utf-8-*-
# Project: CH03
# Filename: perceptron
# Author: DMAN
# Dataset: breast cancer

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import heapq


data = load_breast_cancer()
x = data["data"]
y = data["target"]
number_features = len(x[0])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


class KNN:
    def __init__(self, data, target, k):
        self.data = data
        self.target = target
        self.k = k

    @staticmethod
    def distance(x, y):
        return np.linalg.norm(x - y)

    def predict(self, x):
        result = []
        for y, label in zip(self.data, self.target):
            result.append([self.distance(x, y), label])
        most_k = heapq.nsmallest(self.k, result, lambda x: x[0])

        labels = Counter(np.array(most_k)[:, 1])
        return int(labels.most_common(1)[0][0])

    def test(self, data, target):
        total = 0
        correct = 0
        for x, y in zip(data, target):
            if self.predict(x) == y:
                correct += 1
            total += 1
        return correct / total


class TreeNode:
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.right = None
        self.left = None


class Kd_Tree:
    def __init__(self, data, target):
        dimension = data.shape[1]
        data = np.concatenate((data, target.reshape(-1,1)), axis=1)
        sort_dimension = 0
        self.root = self.build(data, dimension, sort_dimension)

    def build(self, data, d, k):
        if len(data) == 0:
            return None
        elif len(data) == 1:
            return TreeNode(data[0][:-1], data[0][-1])
        else:
            data = sorted(data, key=lambda x: x[k])
            mid = len(data) // 2
            root = TreeNode(data[mid][:-1], data[mid][-1])
            root.left = self.build(data[:mid], d, (k + 1) % d)
            root.right = self.build(data[mid + 1 :], d, (k + 1) % d)
        return root

    def MinNode(self, node, data):
        distance = KNN.distance(node.data, data)
        if distance < self.min_distance:
            self.min_distance = distance
            self.min_target = node.target

    def find(self, root, d, k, data):
        if root.data[k] < data[k]:
            if root.right != None:
                self.find(root.right, d, (k + 1) % d, data)
                # case a
                self.MinNode(root, data)

                # case b
                if root.left != None:
                    distance = KNN.distance(root.left.data, data)
                    if distance < self.min_distance:
                        self.MinNode(root.left, data)
                        self.find(root.left, d, (k + 1) % d, data)
            else:
                self.MinNode(root, data)

        else:
            if root.left != None:
                self.find(root.left, d, (k + 1) % d, data)
                # case a
                self.MinNode(root, data)

                # case b
                if root.right != None:
                    distance = KNN.distance(root.right.data, data)
                    if distance < self.min_distance:
                        self.MinNode(root.right, data)
                        self.find(root.right, d, (k + 1) % d, data)
            else:
                self.MinNode(root, data)

    def search(self, data):
        self.min_distance = KNN.distance(data, self.root.data)
        self.min_target = self.root.target

        # search begin
        self.find(self.root, len(data), 0, data)
        return self.min_target

    def test(self, testdata, testtarget):
        total = 0
        correct = 0
        for x, y in zip(testdata, testtarget):
            total += 1
            if self.search(x) == y:
                correct += 1
        return correct / total


if __name__ == "__main__":
    # model = KNN(x_train, y_train, 7)
    # print("Accuracy : {:.2f}%".format(100 * model.test(x_test, y_test)))
    model = Kd_Tree(x_train, y_train)
    print("Accuracy : {:.2f}%".format(100 * model.test(x_test, y_test)))
