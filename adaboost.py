from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

data = load_breast_cancer()

# check the influence of mean radius
X_train, X_test, y_train, y_test = train_test_split(data.data[:, 0], data.target, test_size=0.2)

y_train = np.array([1 if x==0 else -1 for x in y_train])
y_test = np.array([1 if x==0 else -1 for x in y_test])

class Base_classifier():
    """
    Threshold classifier
    """

    def __init__(self, data, target):
        '''
        :param data: numpy.ndarray
        :param target: numpy.ndarray
        '''
        self._data = data
        self._target = target
        self._best_w = data[0]
        self._check_flag = 0

    def train(self):
        best_accuarcy = 0
        for x in self._data:
            w = x
            outputs_0 = [1 if w>data else -1 for data in self._data]
            outputs_1 = [-1 if w>data else 1 for data in self._data]
            accuracy_0 = accuracy_score(self._target, outputs_0)
            accuracy_1 = accuracy_score(self._target, outputs_1)

            if accuracy_0 > best_accuarcy:
                self._best_w = w
                best_accuarcy = accuracy_0
                self._check_flag = 0

            if accuracy_1 > best_accuarcy:
                self._best_w = w
                best_accuarcy = accuracy_1
                self._check_flag = 1

        print("Best_accuracy in train: {:.2f}%".format(best_accuarcy*100))

    def preidict(self, data):
        if self._check_flag:
            return -1 if self._best_w>data else 1
        else:
            return -1 if self._best_w<data else 1


    def test(self, test_data, test_target):

        if self._check_flag:
            predictions = [-1 if self._best_w>data else 1 for data in test_data]
            score = accuracy_score(test_target, predictions)
        else:
            predictions = [-1 if self._best_w<data else 1 for data in test_data]
            score = accuracy_score(test_target, predictions)

        print("Best_accuracy in test: {:.2f}%".format(score*100))


class Adaboost():

    def __init__(self, x_train, y_train, x_test, y_test):
        self._data = x_train
        self._test_data = x_test
        self._target = y_train
        self._test_target = y_test
        self._init_weight = np.ones(len(x_train))/len(X_train)
        self.M = 10
        self.classifier = []
        self.alpha = []

    def _get_base_classifier(self):

        bcls = Base_classifier(self._data*self._init_weight, self._target)
        bcls.train()
        self.classifier.append(bcls)

        return bcls

    def _sign_(self, x):
        if x<0:
            return -1
        else:
            return 1

    def _cal_e_(self, bcls):

        prediction = np.array([bcls.preidict(x) for x in self._data*self._init_weight])
        print("sum: ", sum(self._target != prediction))
        e = np.array(self._target != prediction) * self._init_weight

        return sum(e)

    def _cal_alpha_(self, e):

        alpha = 1/2*np.log((1-e)/e)
        self.alpha.append(alpha)

    def _cal_Z_(self, bcls, alpha):

        Z_list = np.array([])
        for x, y in zip(self._data, self._target):
            predict = bcls.preidict(x)
            Z_list = np.append(Z_list, np.exp(-alpha *predict*y))
        return sum(Z_list*self._init_weight)

    def _cal_new_weight(self, number):

        bcls = self.classifier[number]
        alpha = self.alpha[number]
        Z = self._cal_Z_(bcls, alpha)
        for idx, w in enumerate(self._init_weight):
            predict = bcls.preidict(self._data[idx])
            y= self._target[idx]
            self._init_weight[idx] = w*np.exp(-alpha*predict*y)/Z


    def train(self):

        for times in range(self.M):
            bcls = self._get_base_classifier()
            e = self._cal_e_(bcls)
            if e == 0:
                self.M = times
                break
            self._cal_alpha_(e)
            self._cal_new_weight(times)

    def test(self):

        score = 0
        total = 0

        for x, y in zip(self._test_data, self._test_target):
            total += 1
            outputs = 0
            for i in range(self.M):
                bcls = self.classifier[i]
                outputs += bcls.preidict(x)*self.alpha[i]
            outputs = self._sign_(outputs)
            if y == outputs:
                score += 1
        print("alpha: ", self.alpha)
        print("Adaboost accuracy is {:.2f}%".format(score*100/total))


if __name__ == '__main__':
    # Weak classifier
    print("show the accuarcy of base classifier")
    bcls = Base_classifier(X_train, y_train)
    bcls.train()
    bcls.test( X_test, y_test)

    # Adaboost
    print("show the accuarcy of Adaboost")
    Ada = Adaboost(X_train,y_train, X_test, y_test)
    Ada.train()
    Ada.test()