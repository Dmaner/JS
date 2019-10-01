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

class SVM:
    def __init__(self, data, label, C=10, max_iter=100):
        self.data = data
        self.label = label
        self.length = len(label)
        self.C = C
        self.iterion = max_iter
        self.alphas = np.zeros(len(data))
        self.b = 0
        self.w = None
        self.err = self.error()

    def f(self, x):
        x = np.mat(x)
        wx = np.mat(self.alphas * self.label) * np.mat(self.data)*x.T
        fx = wx + self.b

        return fx[0, 0]

    def KKT(self):
        """
        7.111-113
        """
        con_1 = self.alphas == 0
        con_2 = (self.alphas < self.C) & (self.alphas > 0)
        con_3 = self.alphas == self.C

        return con_1, con_2, con_3

    def error(self):
        pred = np.dot(np.dot(self.alphas * self.label, self.data), self.data.T)
        error = self.label * pred - 1

        return error

    def select_a1(self, error, ignore):
        """
        reference : https://zhuanlan.zhihu.com/p/27662928
        """
        con_1, con_2, con_3 = self.KKT()
        eligible = ((con_1) & (error > 0)) & \
                   ((con_2) & (error == 0)) & \
                   ((con_3) & (error < 0))
        error[eligible] = 0
        error[ignore] = 0

        # check support vector first
        if error[con_2].any() != 0:
            error1 = error[con_2]
            worst = np.argmax(error1**2)
            a_1 = con_2[worst]
        else:
            a_1 = np.argmax(error**2)
        return a_1

    def select_a2(self, a1, error, ignore):
        error = np.abs(error - error[a1])
        error[ignore] = 0
        return np.argmax(error)

    def update(self, i, j, pair_changed):
        a_i, x_i, y_i = self.alphas[i], self.data[i], self.label[i]
        a_j, x_j, y_j = self.alphas[j], self.data[j], self.label[j]
        fx_i = self.f(x_i)
        fx_j = self.f(x_j)
        E_i = fx_i - y_i
        E_j = fx_j - y_j
        K_ii, K_jj, K_ij = np.dot(x_i, x_i), np.dot(x_j, x_j), np.dot(x_i, x_j)
        eta = K_ii + K_jj - 2 * K_ij
        if eta <= 0:
            print('WARNING  eta <= 0')
            return False
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
            return pair_changed
        self.alphas[i], self.alphas[j] = a_i_new, a_j_new
        b_i = -E_i - y_i * K_ii * (a_i_new - a_i_old) - y_j * K_ij * (a_j_new - a_j_old) + self.b
        b_j = -E_j - y_i * K_ij * (a_i_new - a_i_old) - y_j * K_jj * (a_j_new - a_j_old) + self.b
        self.b = (b_i + b_j) / 2
        self.err = self.error()
        pair_changed += 1

        return pair_changed

    def select_j(self, i):
        j = np.random.randint(0, self.length)
        while j == i:
            j = np.random.randint(0, self.length)
        return j

    def smo(self):
        it = 0
        while it < self.iterion:
            ignore_j = []
            pair_changed = 0
            for i in range(self.length):
                j = self.select_a2(i, self.err, None)
                # j = self.select_j(i)
                pair_changed = self.update(i, j, pair_changed)
            if pair_changed == 0:
                it += 1
            else:
                it = 0
            print('iteration number: {}'.format(it))
        self.w = np.dot(self.alphas * self.label, self.data)


if __name__ == '__main__':
    dataset = load_data()
    data, target = dataset.data, dataset.target
    svm = SVM(data, target)
    svm.smo()
    show_img(data, target, dataset.feature_names, svm.w, svm.b, svm.alphas)


