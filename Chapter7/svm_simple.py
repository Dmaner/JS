# -*-coding:utf-8-*-
# Project: CH07
# Filename: SVM
# Author: DMAN
# Dataset: iris

from sklearn.datasets import load_iris
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    iris = load_iris()
    data, target = iris.data, iris.target
    index = target != 2
    target[target == 0] = -1
    d = namedtuple("dataset", "data target feature_names")
    return d(data[index, :2], target[index], iris.feature_names[:2])


def show_img(data, target, target_names, w, b, alphas):
    xlabel = target_names[0]
    ylabel = target_names[1]
    x = data[:, 0].squeeze()
    y = data[:, 1].squeeze()
    x1 = min(x)
    x2 = max(x)
    a1, a2 = w
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])
    color = list(map(lambda x: "red" if x == 1 else "black", target))
    plt.scatter(x, y, c=color)
    # draw support vector
    spv_x = x[alphas > 0.001]
    spv_y = y[alphas > 0.001]
    plt.scatter(spv_x, spv_y, s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='#AB3319')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


class SimpleSVM:
    def __init__(self, data, label, C=10, max_iter=100):
        self.data = data
        self.label = label
        self.length = len(label)
        self.C = C
        self.iterion = max_iter
        self.alphas = np.zeros(len(data))
        self.b = 0
        self.w = None

    def f(self, x):
        x = np.mat(x).T
        wx = np.mat(self.alphas * self.label)*np.mat(self.data)*x
        fx = wx + self.b
        return fx[0, 0]

    def select_j(self, i):
        j = np.random.randint(0, self.length)
        while j == i:
            j = np.random.randint(0, self.length)
        return j

    def smo(self):
        it = 0
        while it < self.iterion:
            pair_changed = 0
            for i in range(self.length):
                a_i, x_i, y_i = self.alphas[i], self.data[i], self.label[i]
                fx_i = self.f(x_i)
                E_i = fx_i - y_i
                j = self.select_j(i)
                a_j, x_j, y_j = self.alphas[j], self.data[j], self.label[j]
                fx_j = self.f(x_j)
                E_j = fx_j - y_j
                K_ii, K_jj, K_ij = np.dot(x_i, x_i), np.dot(x_j, x_j), np.dot(x_i, x_j)
                eta = K_ii + K_jj - 2 * K_ij
                if eta <= 0:
                    print('WARNING  eta <= 0')
                    continue
                a_i_old, a_j_old = a_i, a_j
                a_j_new = a_j_old + y_j * (E_i - E_j) / eta
                if y_i != y_j:
                    L = max(0, a_j_old - a_i_old)
                    H = min(self.C, self.C + a_j_old - a_i_old)
                else:
                    L = max(0, a_i_old + a_j_old - self.C)
                    H = min(self.C, a_j_old + a_i_old)
                a_j_new = np.clip(a_j_new, L, H)
                a_i_new = a_i_old + y_i * y_j * (a_j_old - a_j_new)
                if abs(a_j_new - a_j_old) < 0.00001:
                    # print('WARNING   alpha_j not moving enough')
                    continue
                self.alphas[i], self.alphas[j] = a_i_new, a_j_new
                b_i = -E_i - y_i * K_ii * (a_i_new - a_i_old) - y_j * K_ij * (a_j_new - a_j_old) + self.b
                b_j = -E_j - y_i * K_ij * (a_i_new - a_i_old) - y_j * K_jj * (a_j_new - a_j_old) + self.b
                self.b = (b_i + b_j) / 2
                pair_changed += 1
                print('INFO   iteration:{}  i:{}  pair_changed:{}'.format(it, i, pair_changed))
            if pair_changed == 0:
                it += 1
            else:
                it = 0
            print('iteration number: {}'.format(it))
        self.w = np.dot(self.alphas * self.label, self.data)


if __name__ == '__main__':
    dataset = load_data()
    data, target = dataset.data, dataset.target
    svm = SimpleSVM(data, target)
    svm.smo()
    show_img(data, target, dataset.feature_names, svm.w, svm.b, svm.alphas)
