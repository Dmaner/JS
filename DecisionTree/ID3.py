from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import math
import pprint as pp

digits = load_digits()
data = digits.data
target = digits.target

# load data
feature_len = data.shape[1]
ent_threshold = 0.1
img_threshold = 4
x_train, x_test, y_train,y_test = train_test_split(data,target,test_size=0.05)

def preprocess(images):
    '''
    二值化处理
    '''
    for idx, image in enumerate(images):
        images[idx] = np.array([0 if x>img_threshold else 1 for x in image])

# preprocess data
preprocess(x_train[:])
preprocess(x_test)

# show some preprocessed data
# plt.imshow(x_train[0].reshape(8,8), cmap="gray")
# plt.title(y_train[0])
# plt.show()

class DT_ID3():
    '''
    ID3算法实现决策树
    '''
    def __init__(self, x_train, y_train, num_feature=64, num_classes=10):
        self.x = x_train
        self.y = y_train
        self.num_classes = num_classes
        self.total_features = num_feature
        self.get_the_ID_3_Tree()

    def cal_ent(self, D):
        '''
        :param D: ndarray
        :param k: ndarray
        '''
        HD = 0
        k_set = set(D)
        total_enties = len(D)
        for k in k_set:
            k_count = len(D[D == k])
            if k_count:
                C_D = (k_count/total_enties)*math.log(k_count/total_enties,2)
                HD -= C_D

        return HD

    def cal_condition_ent(self, feature_col, y_train):

        total_labels = len(y_train)
        pos_index = feature_col == 1
        neg_index = feature_col == 0
        neg_H_DA = (len(feature_col[pos_index]) / total_labels) * self.cal_ent(y_train[pos_index])+ \
                   (len(feature_col[neg_index]) / total_labels) * self.cal_ent(y_train[neg_index])

        return neg_H_DA

    def information_gain(self,feature_col, y_train):

        return self.cal_condition_ent(feature_col, y_train)+self.cal_ent(y_train)

    def empty_subset_check(self,feature_col,y_train):

        pos_index = feature_col == 1
        neg_index = feature_col == 0

        if len(y_train[pos_index]) == len(y_train) or len(y_train[neg_index]) == len(y_train):
            return False

        return True

    def find_max_feature(self, x_train, feature_set, y_train):
        best_feature = feature_set[1]
        best_value = self.information_gain(x_train[:,best_feature], y_train)
        best_index = 1
        for idx, feature in enumerate(feature_set):
            feature_col = x_train[:, feature]
            new_value = self.information_gain(feature_col, y_train)
            if new_value > best_value and self.empty_subset_check(feature_col, y_train):
                best_value = new_value
                best_feature = feature
                best_index = idx

        new_feature_set = np.delete(feature_set, best_index)
        return new_feature_set, best_feature, best_value

    def find_max_label(self, y_train):

        dict = {}
        for item in y_train:
            if not item in dict:
                dict[item] = 1
            else:
                dict[item] +=1

        return sorted(dict.items(), key=lambda item: item[1], reverse=True)[0][0]

    def build_tree(self, x_train, y_train, feature_set):

        node = {}

        if len(y_train) == 0:
            return None

        if len(y_train) == len(y_train[y_train == y_train[0]]):
            node['class'] = y_train[0]
            node['name'] = 'leaf'
            return node

        if len(feature_set) == 1:
            node['class'] = self.find_max_label(y_train)
            node['name'] = 'leaf'
            return node

        new_feature_set, best_feature, best_value = self.find_max_feature(x_train, feature_set,y_train)
        if best_value < ent_threshold:
            node['class'] = self.find_max_label(y_train)
            node['name'] = 'leaf'
            return node

        feature_col = x_train[:,best_feature]
        pos_index = feature_col == 1
        neg_index = feature_col == 0
        node_one = self.build_tree(x_train[pos_index], y_train[pos_index], new_feature_set)
        node_zero = self.build_tree(x_train[neg_index], y_train[neg_index], new_feature_set)
        if node_one and node_zero:
            node['best_feature'] = best_feature
            node['name'] = 'branch'
            node['one'] = node_one
            node['zero'] = node_zero
        else:
            node['name'] = 'leaf'
            node['class'] = self.find_max_label(y_train)

        return node

    def get_the_ID_3_Tree(self):
        feature_set = np.arange(self.total_features)
        self.DT = self.build_tree(self.x,self.y,feature_set)

    def predict(self, samples):
        DT = self.DT
        for i in range(feature_len):
            if DT['name'] == 'leaf':
                return DT['class']
            best_feature = DT['best_feature']
            if samples[best_feature]==1:
                if DT['name']=='leaf':
                    return DT['class']
                elif DT['name'] == 'branch':
                    DT = DT['one']
                else:
                    return "SORRY_IT'S None"
            else:
                if DT['name'] == 'leaf':
                    return DT['class']
                elif DT['name'] == 'branch':
                    DT = DT['zero']
                else:
                    return "SORRY_IT'S None"


model = DT_ID3(x_train, y_train)
count = len(x_test)
correct = 0
for idx, sample in enumerate(x_test):
    real_label = y_test[idx]
    pred = model.predict(sample)
    if real_label == pred:
        correct+=1
print("accuracy: {:.2f}%".format(correct/count*100))
