from Chapter5 import ID3
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from time import clock

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

x_train, x_test, y_train,y_test = train_test_split(x, y, test_size=0.3)

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
def testid3():
    model = ID3.ID3(x_train, y_train)
    print("ID3 Finish training")
    print("Accuracy: {:.2f}%".format(100*model.test(x_test, y_test)))

if __name__ == '__main__':
    testid3()

