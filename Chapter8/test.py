from Chapter8 import boosting_tree
import numpy as np

def load_8_1(filename):
    samples, labels = [], []
    with open(filename, "r") as f:
        data = f.read().split("\n")[1:]
        for line in data:
            samples.append(int(line.split(",")[0]))
            labels.append(int(line.split(",")[1]))
    
    return np.array(samples), np.array(labels)


def load_8_2(filename):
    samples, labels = [], []
    with open(filename, "r") as f:
        data = f.read().split("\n")[1:]
        for line in data:
            samples.append(int(line.split(",")[0]))
            labels.append(float(line.split(",")[1]))

    return np.array(samples), np.array(labels)


def test():
    samples, labels = load_8_2("data/data8-2.txt")
    model = boosting_tree.BoostingTree()
    model.fit(samples, labels, 9)
    print("Loss: {:.2f}".format(model.test(samples, labels)))


if __name__ == "__main__":
    test()