from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from time import clock
from Model import ID3, C45

def record(func):
    def decorater(self, *args, **kwargs):
        begin = clock()
        func()
        end = clock()
        print("{} cost time: {}".format(func.__name__, end-begin))

        return func(*args, **kwargs)
    return decorater

digit = load_digits()
x = digit["data"]
y = digit["target"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

def cost(func):
    def decorater(*args, **kwargs):
        begin = clock()
        val = func(*args, **kwargs)
        end = clock()
        name = func.__name__
        print("{} cost {:.2f} sec".format(name, end-begin))
        return val
    return decorater

@cost
def test(x_train, x_test, y_train,y_test, Model):
    model = Model(x_train.copy(), y_train.copy())
    print("{} finish training".format(model))
    print("Accuracy: {:.2f}%".format(100*model.test(x_test.copy(), y_test.copy())))


if __name__ == '__main__':
    test(x_train, x_test, y_train, y_test, ID3.Model)
    test(x_train, x_test, y_train, y_test, C45.Model)
