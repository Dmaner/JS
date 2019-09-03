# -*-coding:utf-8-*-
# Project: CH03
# Filename: decision tree
# Author: DMAN
# Dataset: digit

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import math

digit = load_digits()
x = digit["data"]
y = digit["target"]

x_train, x_test, y_train,y_test = train_test_split(x, y, test_size=0.3)

# def show():
#     images = digit.images
#     from random import randint
#     random_index = randint(0, len(images))
#     plt.imshow(images[random_index], cmap="gray")
#     plt.title(y[random_index])
#     plt.axis("off")
#     plt.show()

# show()

LEAF = "Leaf"
NODE = "Node"

# too slow for using treenode

class TreeNode:
    def __init__(self, name, feature=None, val=None):
        self.NodeName = name
        if name == LEAF and val == None:
            raise Exception("Leaf doesn't have value!")
        elif name == NODE and val != None:
            raise Exception("Node's value should be None!")
        elif name != NODE and name != LEAF:
            raise Exception("Naming exception!")
        else:
            self.feature = feature
            self.val = val
            self.one = None
            self.zero = None


class ID3:
    def __init__(self, data, target):
        data = self.binarization(data)
        self.classes = set(target)
        self.threshold = 0.1
        features = [x for x in range(0, data.shape[1])]
        self.tree = self.build(data, target, features)

    def cal_entropy(self, x):
        """
        calculate empirical entropy
        """
        H_D = 0
        D = len(x)
        for i in self.classes:
            C_k = sum(x==i)
            if C_k:
                H_D -= C_k / D * math.log(C_k / D, 2)

        return H_D

    def cal_con_entropy(self, target, feature_column):
        """
        calculate conditional empirical entropy
        """
        D = len(target)
        ones = feature_column == 0
        zeros = feature_column == 1
        H_D_A = sum(ones) / D * self.cal_entropy(target[ones]) + sum(
            zeros
        ) / D * self.cal_entropy(target[zeros])

        return H_D_A

    def info_gain(self, target, feature_column):
        """
        calculate infomation gain
        """
        return self.cal_entropy(target) - self.cal_con_entropy(target, feature_column)

    def findmaxlabel(self, target):
        counter = Counter(target)

        return int(counter.most_common(1)[0][0])

    def findmaxfeature(self, target, data, features):
        best_feature = features[0]
        index = 0
        best_value = self.info_gain(target, data[:, best_feature])
        for i in range(1, len(features)):
            feature = features[i]
            value = self.info_gain(target, data[:, feature])
            if value > best_value:
                best_feature = feature
                best_value = value
                index = i

        new_features = np.delete(features, index)

        return new_features, best_feature, best_value

    def build(self, data, target, features):
        if len(target) == 0:
            return None
        elif len(set(target)) == 1:
            return TreeNode(LEAF, val=target[0])
        elif len(features) == 0:
            return TreeNode(LEAF, val=self.findmaxlabel(target))
        else:
            new_feature_set, best_feature, best_value = self.findmaxfeature(target, data, features)
            if best_value < self.threshold:
                return TreeNode(LEAF, val = self.findmaxlabel(target))
            feature_col = data[:,best_feature]
            pos_index = feature_col == 1
            neg_index = feature_col == 0
            if len(pos_index) == 0 or len(neg_index) == 0:
                return TreeNode(LEAF, val=self.findmaxlabel(target))
            else:
                node = TreeNode(NODE, feature=best_feature)
                node.zero = self.build(data[neg_index], target[neg_index], new_feature_set)
                node.one = self.build(data[pos_index], target[pos_index], new_feature_set)
                return node

    def predict(self, x):
        node = self.tree
        while node.NodeName == NODE:
            if x[node.feature] == 0:
                node = node.zero
            else:
                node = node.one
        return node.val

    def binarization(self, x):
        img_threshold = 4
        for i in range(len(x)):
            image = x[i]
            image = [0 if x < img_threshold else 1 for x in image]
            x[i] = image

        return x

    def test(self, x, y):
        total = 0
        correct = 0
        for image, target in zip(x, y):
            if target == self.predict(image):
                correct += 1
            total += 1
        return correct / total


if __name__ == "__main__":
    model = ID3(x_train, y_train)
    print("Finish training")
    print("Accuracy: {:.2f}%".format(100*model.test(x_test, y_test)))