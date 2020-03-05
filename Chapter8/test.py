from Chapter8 import adaboost
import numpy as np

def load(filename):
    samples, labels = [], []
    with open(filename, "r") as f:
        data = f.read().split("\n")[1:]
        for line in data:
            samples.append(int(line.split(",")[0]))
            labels.append(int(line.split(",")[1]))
    
    return np.array(samples), np.array(labels)

def test():
    samples, labels = load("data/data8-1.txt")
    model = adaboost.Model()
    model.fit(samples, labels, 7)

if __name__ == "__main__":
    test()