import numpy as np

class RegressionTree():
    """
    Just split in two part
    """
    def cal_loss(self, data, labels, threshold):
        less = np.where(data <= threshold)
        more = np.where(data > threshold)
        c_1 = 0 if len(data[less]) == 0 else np.mean(labels[less])
        c_2 = 0 if len(data[more]) == 0 else np.mean(labels[more])
        ms = sum(np.square(labels[less] - c_1)) + sum(np.square(labels[more] - c_2)) 
        
        return threshold, ms, c_1, c_2

    def fit(self, data, labels):
        best_threshold, min_loss, best_c1, best_c2 = self.cal_loss(data, labels, data[0])
        for i in range(1, len(data)):
            threshold, loss, c1, c2 = self.cal_loss(data, labels, data[i])
            if loss < min_loss:
                best_threshold = threshold
                min_loss = loss
                best_c1 = c1
                best_c2 = c2
        self.threshold = best_threshold
        self.c1 = best_c1
        self.c2 = best_c2

        return min_loss
    
    def predict(self, x):
        return self.c1 if x <= self.threshold else self.c2

    def paramgrams(self):
        return self.threshold, self.c1, self.c2


class BoostingTree():
    def __init__(self):
        self.trees = []

    def fit(self, data, labels, iterition):
        for _ in range(iterition):
            tree = RegressionTree()
            loss = tree.fit(data, labels)
            threshold, c1, c2 = tree.paramgrams()
            print("Threshold: {:.2f}, C_1 {:.2f}, C_2: {:.2f}, Loss {:.2f}".format(threshold, c1, c2, loss))
            print(labels)
            labels = np.array([y - tree.predict(x) for x, y in zip(data, labels)])
            self.trees.append(tree.paramgrams())

    def predict(self, x):
        ans = 0
        for threshold, c1, c2 in self.trees:
            if x <= threshold:
                ans += c1
            else:
                ans += c2
            
        return ans

    def test(self, data, label):
        loss = 0
        for x, y in zip(data, label):
            loss += (y - self.predict(x)) ** 2

        return loss