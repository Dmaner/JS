import numpy as np

LESS = 1
MORE = 0


class WeakClassifier():
    def __init__(self, thresold, symbol):
        self.thresold = thresold
        self.symbol = symbol

    def fit(self, x):
        if self.symbol == LESS:
            return 1 if x <= self.thresold else -1
        else:
            return 1 if x > self.thresold else -1
        
    def test(self, samples, labels):
        return np.array([1 if self.fit(x)!=y else 0 for x, y in  zip(samples, labels)])


class Model():
    def __init__(self):
        self.classifiers = []

    def fit(self, samples, labels, nums_of_classifier):
        self.w = np.ones(len(samples)) / len(samples)
        self.alphas = []
        index = 0
        for _ in range(nums_of_classifier):
            clf_less = WeakClassifier(samples[index], LESS)
            clf_more = WeakClassifier(samples[index], MORE)
            e_less = sum(clf_less.test(samples, labels) * self.w)
            e_more = sum(clf_more.test(samples, labels) * self.w)
            clf = clf_more if e_less > e_more else clf_less
            e = min(e_less, e_more)
            alpha = self.cal_alpha(e)
            self.w = self.update_w(clf, alpha, samples, labels)
            self.classifiers.append(clf)
            self.alphas.append(alpha)
            index += 1
            print("No.{:<2d} classifier error: {:.2f}".format(index, e) )

        # test
        for sample, label in zip(samples, labels):
            ans = 0
            correct = 0
            total = 0
            for i in range(index):
                ans += self.alphas[i] * self.classifiers[i].fit(sample)

            if self.sign(ans) == label:
                correct += 1
            total += 1
        
        print("Accuracy: {:.2f}".format(correct / total))
    
    def sign(self, x):
        return 1 if x > 0 else -1

    def cal_alpha(self, e):
        return np.log((1-e)/e) / 2

    def update_w(self, clf, alpha, samples, labels):
        arr = self.w * np.exp(np.array([-alpha * y *clf.fit(x) for x, y in zip(samples, labels)]))
        # calculate Zm
        Z = sum(arr)

        return arr / Z

    def __repr__(self):
        return "Adaboost"