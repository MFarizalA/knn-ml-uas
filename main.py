import numpy as np
from PIL import Image
import operator
from operator import itemgetter
import os
import numpy as np
from sklearn.metrics import accuracy_score

def euc_dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, K=3):
        self.K = K

    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        count = 0
        for i in range(len(X_test)):
            count = count + 1
            dist = np.array([euc_dist(X_test[i], x_t) for x_t in self.X_train])
            dist_sorted = dist.argsort()[:self.K]
            neigh_count = {}
            for idx in dist_sorted:
                if self.Y_train[idx] in neigh_count:
                    neigh_count[self.Y_train[idx]] += 1
                else:
                    neigh_count[self.Y_train[idx]] = 1
            sorted_neigh_count = sorted(neigh_count.items(), key=operator.itemgetter(1), reverse=True)
            print(str(count) + ' ' + str(sorted_neigh_count[0][0]))
            predictions.append(sorted_neigh_count[0][0])
        return predictions

if __name__ == '__main__':

    img = Image.open(#image directory)
    img = img.resize((128, 128))

    width, height = img.size
    X = []
    y = []
    count = 0
    dir = "/tmp/105_classes_pins_dataset/"
    for i in os.listdir(dir):
        print(i, ":", len(os.listdir(dir + "/" + i)))
        count += len(os.listdir(dir + "/" + i))
        for j in os.listdir(dir + "/" + i):
            img = Image.open(dir + "/" + i + "/" + j)
            img = img.resize((128, 128))
            X.append(np.asarray(img))
            y.append(i)
    print(count)
    X = np.asarray(X)
    y = np.asarray(y)

    X = X.reshape(17534, 49152).astype('float32')

    model = KNN(K=k)
    model.fit(X_test, y_test)
    pred = model.predict(X_test)
