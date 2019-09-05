from . import ID3
from math import log


class Model(ID3.Model):

    def __repr__(self):
        return "C 4.5 algorithm"

    def info_gain(self, target, feature_column):
        return super().info_gain(target, feature_column) / self.cal_entropy(target)

    def cal_entropy_a(self, feature_column):
        """
        calculate features
        """
        H_A = 0
        total = len(feature_column)
        ones = feature_column[feature_column == 1]
        zeros = feature_column[feature_column == 0]

        H_A = - ones/total * log(ones/total, 2) - \
            zeros / total * log(zeros/total, 2)

        return H_A