from .ID3 import *


class RegModel():

    """
    CART for regression
    test dataset:
    """
    def __init__(self, data, target):
        pass

    def find_s(self, col, target, mins, maxs, value):
        best_s = mins
        for s in range(mins, maxs+1):
            lower = col < s
            higher = col > s
            c1 = np.mean(target[lower])
            c2 = np.mean(target[higher])

            result = np.sum(np.square(target-c1)) + np.sum(np.square(target-c2))
            if value == 0 or result < value:
                best_s = s
                value = result
        return best_s, value

    def find_j(self, data, target, features):

        best_feature = None
        best_s = None
        best_value = 0

        for feature in features:
            col = data[:, feature]
            mins, maxs = min(col), max(col)
            s, value = self.find_s(col, target, mins, maxs, value)
            if best_feature == None or value < best_value:
                best_feature = feature
                best_s = s
                best_value = value

        return best_feature, best_s

    def build(self, data, target, features):
        j, s = self.find_j(data, target, features)


class ClaModel(Model):

    def __repr__(self):
        return "Classifier CART"

    # def build(self, data, target, features):
    #     if len(features) == 0 or len(target) < 4:
    #         return TreeNode(LEAF, val=self.findmaxlabel(target))
    #     elif len(set(target)) == 1:
    #         return TreeNode(LEAF, val=target[0])
    #     else:
    #         feature, newfeatures = self.findmaxfeature(target, data, features)
    #         col = data[:, feature]
    #         zeros = col == 0
    #         ones = col == 1
    #         node = TreeNode(NODE, feature=feature)
    #         node.one = self.build(data[ones], target[ones], newfeatures)
    #         node.zero = self.build(data[zeros], target[zeros], newfeatures)
    #
    #         return node

    def Gini(self, target):
        """
        calculate gini index
        :param target: data
        :return: Gini
        """
        r = 1
        total = len(target)
        classes = set(target)
        for c in classes:
            k = len(target[target==c])
            r -= (k/total)**2

        return r

    def ConGini(self, feature_col, target):
        """
        Calculate D's gini index divided by feature
        """
        total = len(target)
        d1 = target[feature_col == 0]
        d2 = target[feature_col == 1]

        return len(d1)/total * self.Gini(d1) + len(d2)/total * self.Gini(d2)

    def findmaxfeature(self, target, data, features):
        """
        find min gini index feature
        """
        feature = features[0]
        value = self.ConGini(data[:, feature], target)
        index = 0
        for i in range(1, len(features)):
            f = features[i]
            v = self.ConGini(data[:, f], target)
            if v < value:
                feature = f
                value = v
                index = i

        newfeatures = np.delete(features, index)

        return newfeatures, feature, value



