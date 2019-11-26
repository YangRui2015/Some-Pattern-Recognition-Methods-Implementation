# -*- coding: utf-8 -*-
# Pattern_Recognition_experiment1.py
# author: yangrui
# description: 
# created: 2019-11-21T16:06:29.997Z+08:00
# last-modified: 2019-11-23T09:08:40.266Z+08:00
# email: yangrui19@mails.tsinghua.edu.cn


import numpy as np 
import sklearn 
from minist_loader import load_train_images, load_train_labels, load_test_images, load_test_labels
import matplotlib.pyplot as plt
import pickle


def mean(array, axis=None):
    if axis is None:
        return array.sum() / (array.shape[0] * array.shape[1])
    else:
        return array.sum(axis=axis) / array.shape[axis]

def var(array, axis=None):
    if axis is None:
        return np.power(array - mean(array), 2).sum() / (array.shape[0] * array.shape[1])
    elif axis == 0:
        return np.power(array - mean(array, axis=axis), 2).sum(axis=axis) / array.shape[axis]
    elif axis == 1:
        return np.power(array - mean(array, axis=axis).reshape(-1,1), 2).sum(axis=axis) / array.shape[axis]
    else:
        raise NotImplementedError

def cov(array):
    return array.T.dot(array) / array.shape[0]



def PCA(X, k):
    assert len(X.shape) == 2, 'Wrong shape of input'
    X_mean = mean(X, axis=0)
    X_diff_mean = X - X_mean
    X_cov = X_diff_mean.T.dot(X_diff_mean) / (X.shape[0] - 1)

    eig_vals, eig_vecs = np.linalg.eig(X_cov)

    # eig_vecs[:, np.where(eig_vals < 0)] = - eig_vecs[:, np.where(eig_vals < 0) ]
    # eig_vals = np.abs(eig_vals)

    sort_index = np.argsort(- np.abs(eig_vals))
    eig_vals_sorted = eig_vals[sort_index]

    total_eig_val = sum(eig_vals)
    var_ratio = eig_vals_sorted / total_eig_val
    # print('var:{}\n sum ratio:{}'.format(eig_vals_sorted[:k], var_ratio[:k].sum())

    sort_index = sort_index[:k]
    eig_vals_sorted = eig_vals[sort_index]
    eig_vecs_sorted = eig_vecs[:, sort_index]

    return X_diff_mean.dot(eig_vecs_sorted)


def gaussion_prob(x, mu, var):  # 
    prob = 1 / np.sqrt(2 * np.pi * var) * np.exp(- np.power((x - mu), 2) / (2 * var))
    return np.prod(prob, axis=1)

def gaussion_prob_matrix(x, mu, var):  # 
    d = x.shape[1]
    assert var.shape[0] == var.shape[1], 'Shape not the same!'
    det = np.linalg.det(var)
    inv = np.linalg.inv(var)
    return 1 / np.sqrt(np.power(2 * np.pi, d) * det) * np.exp(- 1 / 2 *  ((x - mu).T * np.dot(inv, (x-mu).T)).sum(axis=0)) 


class BayesClassifier:
    def __init__(self, num_classes=10, independent=False):
        self.mu = None
        self.var = None
        self.num_classes = num_classes
        self.independent = independent
    
    def fit(self, train, train_label):
        self.mu = []
        self.var = []

        for i in range(self.num_classes):
            index = np.where(train_label == i)
            temp = train[index]
            temp_mu = mean(temp, axis=0)

            if self.independent:
                temp_var = var(temp, axis=0)
            else:
                temp_var = cov(temp)

            self.mu.append(temp_mu)
            self.var.append(temp_var)

            # print(temp_mu, temp_var)
            
    def predict(self, X):
        probs = []
        for i in range(self.num_classes):
            temp_mu, temp_var = self.mu[i], self.var[i]
            if self.independent:
                prob = gaussion_prob(X, temp_mu, temp_var)
            else:
                prob = gaussion_prob_matrix(X, temp_mu, temp_var) 

            probs.append(prob)
        probs_v = np.vstack(probs)
        pred = np.argmax(probs_v, axis=0)
        return pred
    
    def fit_predict(self, X, label):
        self.fit(X, label)
        return self.predict(X)

    def score(self, X, label, fit=True):
        if fit:
            return (self.fit_predict(X, label) == label).sum() / X.shape[0]
        else:
            return (self.predict(X) == label).sum() / X.shape[0]


if __name__ == "__main__":
    img_train = load_train_images()
    train_label = load_train_labels()
    img_test = load_test_images()
    test_label = load_test_labels()
    print(img_train.shape, img_test.shape)

    img_train_flatten = img_train.reshape(img_train.shape[0], img_train.shape[1] * img_train.shape[2])
    img_test_flatten = img_test.reshape(img_test.shape[0], img_test.shape[1] * img_test.shape[2])
    img_total_flatten = np.vstack((img_train_flatten, img_test_flatten))


    # # 二维可视化
    # k = 2
    # img_train_project = PCA(img_train_flatten , k=k)[:1000]
    # temp_label = train_label[:1000]
    # for i in range(10):
    #     index = np.where(temp_label==i)
    #     plt.scatter(img_train_project[index, 0], img_train_project[index, 1])
    # plt.show()
    # import pdb; pdb.set_trace()


    train_scores = []
    test_scores = []
    k_list =  [1]  + list(range(5, 101, 5))  # + list(range(10, 201, 10)) 
    for i in k_list:
        print(i)
        img_total_project = PCA(img_total_flatten, k=i)
        img_train_project = img_total_project[:img_train.shape[0]]
        img_test_project = img_total_project[img_train.shape[0]:]
 
        bayes = BayesClassifier(independent=False)
        train_accuracy = bayes.score(img_train_project, train_label)
        test_accuracy = bayes.score(img_test_project, test_label, fit=False)
        print('train accuracy:{}, test accuracy:{}'.format(train_accuracy, test_accuracy))

        train_scores.append(train_accuracy)
        test_scores.append(test_accuracy)

    plt.figure()
    plt.plot(k_list, train_scores, label='train_accuracy')
    plt.plot(k_list, test_scores, label='test accuracy')
    plt.legend()
    plt.show()

    # import pdb; pdb.set_trace()

































