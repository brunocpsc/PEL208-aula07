# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 23:00:12 2019
LMS,LDA,PCA e SVM
"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm

def RL():
    iris = load_iris()
    X = iris.data
    y = iris.target
    i=0
    
    #retirada da classe 2 (Virginica) do dataset
    while (y[i]!=2):
        i=i+1
    X=X[:i]
    y=y[:i]
    
    X0=X[:,[0]]
    X1=X[:,[1]]
    X2=X[:,[2]]
    X3=X[:,[3]]
    
    def fazRL(x,y):
    
        reg=LinearRegression().fit(x,y)
        reg.score(x,y)
        y_pred=reg.predict(x)
        #print('Score RL:', reg.score(x,y))
        
        plt.scatter(x, y,  color='green', label='IRIS DS')
        plt.plot(x,y_pred, color='blue', linewidth=3, label='RL')
        plt.legend()
        plt.title('RL')
        plt.grid()
        plt.show()
    
    fazRL(X0,y)
    fazRL(X1,y)
    fazRL(X2,y)
    fazRL(X3,y)
    
    return 1

def PCA_LDA():
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    
    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)
    #print('PCA Score:', pca.score(X,y))
    
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)
    #print('LDA Score:', lda.score(X,y))
    
    plt.figure()
    colors = ['blue', 'green', 'red']
    lw = 2
    
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend()
    plt.title('PCA')
    plt.grid()
    
    plt.figure()
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                    label=target_name)
    plt.legend()
    plt.title('LDA')
    
    plt.grid()
    plt.show()
    
    return 1
    

def SVM():
    # import some data to play with
    iris = load_iris()
    # Take the first two features. We could avoid this by using a two-dim dataset
    X = iris.data[:, :2]
    y = iris.target
    
    clf = (svm.SVC(kernel='rbf', gamma=0.7, C=1.0))
    clf.fit(X, y)
    
    X0, X1 = X[:, 0], X[:, 1]
    colors = ['blue', 'green', 'red']
    target_names = iris.target_names
    
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X0[y == i], X1[y == i], alpha=.8, color=color,
                    label=target_name)
    plt.legend()
    plt.title('SVM')
    
    plt.grid()
    plt.show()
    

RL()
PCA_LDA()
SVM()
    
    
    