# -*- coding: utf-8 -*-
# Pattern_Recognition_experiment2.py
# author: yangrui
# description: 
# created: 2019-11-23T09:16:43.621Z+08:00
# last-modified: 2019-11-23T09:16:43.621Z+08:00
# email: yangrui19@mails.tsinghua.edu.cn


import numpy as np 
import sklearn 
from minist_loader import load_train_images, load_train_labels, load_test_images, load_test_labels
import matplotlib.pyplot as plt 
from Pattern_Recognition_experiment1 import PCA

img_train = load_train_images() / 255
train_label = load_train_labels()
img_test = load_test_images()  / 255
test_label = load_test_labels()
print(img_train.shape, img_test.shape)


def plot_two_class(kk, bb):    #可视化两类数据图片
    index_kk = np.where(train_label == kk)[0][0]
    index_bb = np.where(train_label == bb)[0][0]
    plt.figure()
    plt.subplot(121)
    plt.imshow(img_train[index_kk] * 255,cmap='gray')
    plt.title(f'class {kk}')
    plt.subplot(122)
    plt.imshow(img_train[index_bb] * 255,cmap='gray')
    plt.title(f'class {bb}')
    plt.show()

# plot_two_class(7,9)
# import pdb; pdb.set_trace()


img_train_flatten = img_train.reshape(img_train.shape[0], img_train.shape[1] * img_train.shape[2])
img_test_flatten = img_test.reshape(img_test.shape[0], img_test.shape[1] * img_test.shape[2])
img_total_flatten = np.vstack((img_train_flatten, img_test_flatten))


img_total_project = PCA(img_total_flatten, k=50)
img_train_project = img_total_project[:img_train.shape[0]]
img_test_project = img_total_project[img_train.shape[0]:]

class Perceptron:
    def __init__(self):
        self.w = None
        self.b = None
        self.min_delta = 0.0001
        self.max_iters = 5000
        self.lr = 0.02
    
    def loss_func(self, data, label):
        assert self.w is not None and self.b is not None, "w,b can't be none !"
        return 0.5 * np.power(label.reshape(-1) - self.predict(data), 2).sum()

    def fit(self, train, train_label):
        self.w, self.b = np.zeros((1,train.shape[1]), dtype=np.float32), np.zeros(1, np.float32)
        self.iters = 0
        loss = self.loss_func(train, train_label)

        while self.iters <= self.max_iters:
            # print('iters:{}'.format(self.iters))
            pred = self.predict(train)

            self.w += self.lr * ((train_label.reshape(-1) - pred).reshape(-1,1) * train).mean(axis=0)
            self.b += self.lr * (train_label.reshape(-1) - pred).mean()

            loss_new = self.loss_func(train, train_label)
            # print('loss:{}'.format(loss))

            self.iters += 1

            if abs(loss_new - loss) < self.min_delta:
                print('training for {} iters, loss {}'.format(self.iters, loss_new))
                break
            loss = loss_new

            
    def predict(self, X):
        assert self.w is not None and self.b is not None, "w,b can't be none !"
        return (self.w * X + self.b).sum(axis=1)

    def fit_predict(self, X, label):
        self.fit(X, label)
        return self.predict(X)
    
    def score(self, X, label, fit=True):
        if fit:
            return (np.round(self.fit_predict(X, label)) == label.reshape(-1)).sum() / X.shape[0]
        else:
            return (np.round(self.predict(X)) == label.reshape(-1)).sum() / X.shape[0]

    
def get_class(k, img_train=img_train_project, train_label=train_label, img_test=img_test_project, test_label=test_label):
    train_index = np.where(train_label == k)
    test_index = np.where(test_label == k)
    return img_train[train_index], train_label[train_index], img_test[test_index], test_label[test_index]


def get_two_clsses(k1, k2, img_train=img_train_project, train_label=train_label, img_test=img_test_project, test_label=test_label):
    train_a, train_label_a, test_a, test_label_a = get_class(k1)
    train_b, train_label_b, test_b, test_label_b = get_class(k2)

    train_label_a[:] = 0
    test_label_a[:] = 0
    train_label_b[:] = 1
    test_label_b[:] = 1

    train_merge = np.vstack((train_a, train_b))
    train_label_merge = np.vstack((train_label_a.reshape(-1,1), train_label_b.reshape(-1,1)))
    test_merge = np.vstack((test_a, test_b))
    test_label_merge = np.vstack((test_label_a.reshape(-1,1), test_label_b.reshape(-1,1)))

    return train_merge, train_label_merge, test_merge, test_label_merge

def experiment(k0, k1):
    print('\nclass {} & {}'.format(k0, k1))
    train_merge, train_label_merge, test_merge, test_label_merge = get_two_clsses(k0, k1)

    train_merge = train_merge.astype(np.float32) 
    train_label_merge = train_label_merge.astype(np.float32)
    test_merge = test_merge.astype(np.float32) 
    test_label_merge = test_label_merge.astype(np.float32)

    classifier = Perceptron()
    train_acc = classifier.score(train_merge, train_label_merge)
    test_acc = classifier.score(test_merge, test_label_merge, fit=False)

    print('train accuracy:{}, test accuracy:{} '.format(train_acc, test_acc))


for i in range(10):
    for j in range(i+1, 10):
        experiment(i,j)


import pdb; pdb.set_trace()






