from sklearn import datasets
import math
import numpy as np
from sklearn.model_selection import train_test_split

#Load data
data = datasets.load_iris()
names = data.target_names
x = data.data
y = data.target

# split trainset and testset
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2)
print(x_test.shape, x_train.shape)

# show some samples
y_train_c = np.array([y_train])
y_train_c = y_train_c.T
# print(np.concatenate((x_train, y_train_c), axis=1))

def shuffle_data(X, y, seed=None):
    if seed:
        np.random.seed(seed)

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    return X[idx], y[idx]


# 正规化数据集 X
def normalize(X, axis=-1, p=2):
    lp_norm = np.atleast_1d(np.linalg.norm(X, p, axis))
    lp_norm[lp_norm == 0] = 1
    return X / np.expand_dims(lp_norm, axis)


# 标准化数据集 X
def standardize(X):
    X_std = np.zeros(X.shape)
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    # 做除法运算时请永远记住分母不能等于0的情形
    # X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
    return X_std


# 划分数据集为训练集和测试集
def train_test_split(X, y, test_size=0.2, shuffle=True, seed=None):
    if shuffle:
        X, y = shuffle_data(X, y, seed)

    n_train_samples = int(X.shape[0] * (1 - test_size))
    x_train, x_test = X[:n_train_samples], X[n_train_samples:]
    y_train, y_test = y[:n_train_samples], y[n_train_samples:]

    return x_train, x_test, y_train, y_test


def accuracy(y, y_pred):
    y = y.reshape(y.shape[0], -1)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    return np.sum(y == y_pred) / len(y)


class NaiveBayes():
    """朴素贝叶斯分类模型. """

    def __init__(self,X,y):
        self.classes = np.unique(y)
        self.X = X
        self.y = y
        self.features = X.shape[1]
        # 存储高斯分布的参数(均值, 方差), 因为预测的时候需要, 模型训练的过程中其实就是计算出
        # 所有高斯分布(因为朴素贝叶斯模型假设每个类别的样本集每个特征都服从高斯分布, 固有多个
        # 高斯分布)的参数
        self.parameters = {}

    def Get_mean_and_val(self):
        # 计算每一个类别每个特征的均值和方差
        for feature in range(self.features):
            feature_col = self.X[:, feature]
            for label in range(len(self.classes)):
                mean = feature_col[self.y == label].mean()
                val = feature_col[self.y == label].var()
                self.parameters[(feature,label)] = (mean, val)

    # 计算高斯分布密度函数的值
    def calculate_gaussian_probability(self, mean, var, x):
        coeff = (1.0 / (math.sqrt((2.0 * math.pi) * var)))
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var)))
        return coeff * exponent

    # 计算先验概率
    def calculate_priori_probability(self):
        y_probability = np.zeros(len(self.classes))
        for i in range(len(self.classes)):
            result = self.X[self.y == i].shape[0]/self.X.shape[0]
            y_probability[i] = result
        return np.array(y_probability)

    # Classify using Bayes Rule, P(Y|X) = P(X|Y)*P(Y)/P(X)
    # P(X|Y) - Probability. Gaussian distribution (given by calculate_probability)
    # P(Y) - Prior (given by calculate_prior)
    # P(X) - Scales the posterior to the range 0 - 1 (ignored)
    # Classify the sample as the class that results in the largest P(Y|X)
    # (posterior)
    def classify(self, sample):
        self.Get_mean_and_val()
        posteriors = []

        # 遍历所有类别
        prior_list = self.calculate_priori_probability()
        for i in range(len(self.classes)):
            prior = prior_list[i]
            posterior = np.log(prior)

            # probability = P(Y)*P(x1|Y)*P(x2|Y)*...*P(xN|Y)
            # 遍历所有特征
            for j in range(self.features):
                # 取出第i个类别第j个特征的均值和方差
                mean,var = self.parameters[(j,i)]
                # 取出预测样本的第j个特征
                sample_feature = sample[j]
                # 按照高斯分布的密度函数计算密度值
                prob = self.calculate_gaussian_probability(mean, var, sample_feature)
                # 朴素贝叶斯模型假设特征之间条件独立，即P(x1,x2,x3|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y),
                # 并且用取对数的方法将累乘转成累加的形式
                posterior += np.log(prob)

            posteriors.append(posterior)

        # 对概率进行排序
        return np.argmax(posteriors)

    # 对数据集进行类别预测
    def predict(self, X):
        y_pred = []
        for sample in X:
            y = self.classify(sample)
            y_pred.append(y)
        return np.array(y_pred)

clf = NaiveBayes(x_train, y_train)
y_pred = np.array(clf.predict(x_test))
accu = accuracy(y_test, y_pred)
print("Accuracy:", accu)
