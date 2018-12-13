import numpy as np

name = 'rbf'
sigma = 1.3
C = 200
toler = 0.0001
itertion = 10000

def loadDataSet(filename):
    x = []
    y = []
    with open(filename, 'r') as f:
        oneline = f.readline()[:-1].split()
        while oneline:
            x.append([float(oneline[0]), float(oneline[1])])
            y.append(int(float(oneline[2])))
            oneline = f.readline()[:-1].split()

    return np.array(x), np.array(y)

def selectJrand(i,m): #在0-m中随机选择一个不是i的整数
    j=i
    while (j==i):
        j=int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):  #保证a在L和H范围内（L <= a <= H）
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

def build_kernel(xi, xj, kernel_name, weight = None):
    m, n = xi.shape
    K = np.mat(np.zeros((m, 1)))
    if kernel_name == 'lin':
        K = xi*xj.T
    elif kernel_name == 'rbf':
        for i in range(m):
            deltaRow = xi[i,:] - xj
            K[i] = deltaRow*deltaRow.T
        K = np.exp(K/(-1*weight**2))
    else:
        raise NameError("NO kernel is type of {}".format(kernel_name))

    return K

class SVM():

    def __init__(self, x_train, y_train, C, toler, kernel_name, weight=None):
        self.x = x_train
        self.y = y_train
        self.C = C
        self.toler = toler
        self.weight = weight
        self.m = x_train.shape[0]
        self.cache = np.mat(np.zeros((self.m, 2)))
        self.b = 0
        self.K = np.mat(np.zeros((self.m, self.m)))
        self.alphas = np.mat(np.zeros((self.m, 1)))
        for i in range(self.m):
            self.K[:, i] = build_kernel(self.x, self.x[i, :], kernel_name, weight)

    def cal_Ei(self, i):
        '''
        P127 公式7.105
        '''
        g_xi = float(np.multiply(self.alphas, self.y).T * self.K[:, i] + self.b)
        Ei = g_xi - float(self.y[i])

        return Ei

    def select_J(self, i, Ei):
        '''
        P129 第二个变量的选择
        '''
        J_max_index = -1
        delta_E_max = 0
        Ej = 0
        self.cache[i] = [1, Ei]
        vaild_cache_list = np.nonzero(self.cache[:,0].A)[0]
        if len(vaild_cache_list) >1:
            for k in vaild_cache_list:
                if k == i:
                    continue
                Ek = self.cal_Ei(k)
                delta_E = np.abs(Ei - Ek)
                if delta_E_max < delta_E:
                    delta_E_max = delta_E
                    J_max_index = k
                    Ej = Ek
            return J_max_index, Ej
        else:
            j = selectJrand(i, self.m)
            Ej = self.cal_Ei(j)
            return j, Ej

    def update(self, k):
        Ek = self.cal_Ei(k)
        self.cache[k] = [1 ,Ek]

    def updata_weight(self, i):
        '''
        参数更新公式P125~P128
        '''
        Ei = self.cal_Ei(i)
        if ((self.y[i] * Ei < -self.toler) and (self.alphas[i] < self.C)) or (
                (self.y[i] * Ei > self.toler) and (self.alphas[i] > 0)):
            j ,Ej = self.select_J(i, Ei)
            a_old_i = self.alphas[i].copy()
            a_old_j = self.alphas[j].copy()
            if self.y[i] != self.y[j]:
                L = max(0, self.alphas[j] - self.alphas[i])
                H = min(self.C, self.C+self.alphas[j] - self.alphas[i])
            else:
                L = max(0, self.alphas[j]+self.alphas[i]-self.C)
                H = min(self.C, self.alphas[j]+self.alphas[i])
            if L==H:
                print("L == H")
                return 0
            eta = 2.0*self.K[i,j] - self.K[i,i] - self.K[j, j]
            if eta >= 0:
                print("eta>=0")
                return 0
            self.alphas[j] -= self.y[j]*(Ei-Ej)/eta
            self.alphas[j] = clipAlpha(self.alphas[j], L=L, H=H)
            self.update(j)
            if np.abs((self.alphas[j]-a_old_j) < self.toler):
                print("j not moving enough")
                return 0
            self.alphas[i] += self.y[j]*self.y[i]*(a_old_j-self.alphas[j])
            self.update(i)
            b1 = self.b - Ei - self.y[i]*(self.alphas[i]-a_old_i)*self.K[i, i] - self.y[j]*(self.alphas[j]-a_old_j)*self.K[i,j]
            b2 = self.b - Ej - self.y[i]*(self.alphas[i]-a_old_i)*self.K[i, j] - self.y[j]*(self.alphas[j]-a_old_j)*self.K[j,j]
            if 0<self.alphas[i]<self.C:
                self.b = b1
            elif 0<self.alphas[j]<self.C:
                self.b = b2
            else:
                self.b = (b1+b2)/2.0
            return 1
        else:
            return 0

    def smo(self, maxIter):
        iter = 0
        entireSet = True
        alphaPairsChanged = 0
        while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
            alphaPairsChanged = 0
            if entireSet:
                for i in range(self.m):
                    alphaPairsChanged += self.updata_weight(i)
                    print("fullest, iter: {},i: {} pairs changed {}".format(iter, i, alphaPairsChanged))
                iter += 1
            else:
                nonBoundIs = np.nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
                for i in nonBoundIs: #遍历非边界的数据
                    alphaPairsChanged += self.updata_weight(i)
                    print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
                iter += 1
            if entireSet:
                entireSet = False
            elif (alphaPairsChanged == 0):
                entireSet = True

        return self.b, self.alphas

def testrbf(train_path, test_path, C, toler, kernel_name, maxIter, weight=None):
    x_train, y_train = loadDataSet(train_path)
    x_test, y_test = loadDataSet(test_path)
    print(np.mat(x_train).shape, np.mat(y_train).transpose().shape)
    dataMat = np.mat(x_train)
    labelMat = np.mat(y_train).transpose()
    svm = SVM(dataMat, labelMat, C, toler, kernel_name, weight)
    b, alphas = svm.smo(maxIter)
    svIndex = np.nonzero(alphas)[0]
    sVs = dataMat[svIndex]
    labelsv = labelMat[svIndex]
    print("SVS", sVs)
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    m, n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernel = build_kernel(sVs, dataMat[i, :],kernel_name, weight)
        prediction = kernel.T*np.multiply(labelsv, alphas[svIndex]) + b
        if np.sign(prediction)!=np.sign(y_train[i]):
            errorCount += 1
    print("Training error: {:.3f}".format(float(errorCount)/m))
    errorCount_test = 0
    dataMat_test = np.mat(x_test)
    m, n = np.shape(dataMat_test)
    for i in range(m):
        kernel = build_kernel(sVs, dataMat_test[i,:],'rbf', 1.3)
        prediction = kernel.T*np.multiply(labelsv, alphas[svIndex]) + b
        if np.sign(prediction) != np.sign(y_test[i]):
            errorCount_test += 1
    print("The test error rate is {:.3f}".format(float(errorCount_test)/m))

def main():
    filename_traindata='testdata.txt'
    filename_testdata='testdata.txt'
    testrbf(filename_traindata, filename_testdata, C, toler, 'rbf', itertion, 1.3)

if __name__ == '__main__':
    main()
