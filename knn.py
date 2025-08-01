#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math


# In[2]:


dataset = pd.read_csv('/Users/saumyapandey/Machine-Learning-Models/datasets/wdbc.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[3]:


dataset.head()


# In[4]:


def normalize_data(X):
    # this normalization method uses z-score for normalization of dataset
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)

    return (X-mean_X)/std_X


# In[5]:


def euclidean_distance(dp1, dp2):
    dist = 0.0
    for i in range(len(dp1)):
        dist += pow((float(dp1[i]) - float(dp2[i])), 2)
    return math.sqrt(dist)                    


# In[6]:


def k_nearest_neighbours(X_train, y_train, test_point, k):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(test_point, X_train[i])
        distances.append((dist, y_train[i]))
    distances.sort();
    k_nearest = distances[:k]
    
    count_label = {}
    for _, label in k_nearest:
        if label in count_label:
            count_label[label] += 1
        else:
            count_label[label] = 1
        
    return max(count_label, key=count_label.get)


# In[7]:


def evaluate_knn(train_X, train_y, test_X, test_y, k):
    correct = 0
    for i in range(len(test_X)):
        predicted_label = k_nearest_neighbours(train_X, train_y, test_X[i], k)
        if predicted_label == test_y[i]:
            correct += 1  
    return correct / len(test_y)  


# In[8]:


def main():
    k_values = list(range(1, 53, 2))
    
    train_accuracies = {k: [] for k in k_values}  
    test_accuracies = {k: [] for k in k_values} 
    
    for _ in range(20):  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True)

        if normalize_data:
            X_train = normalize_data(X_train)    
            X_test = normalize_data(X_test)  
        
        for k in k_values:
            train_acc = evaluate_knn(X_train, y_train, X_train, y_train, k)  
            test_acc = evaluate_knn(X_train, y_train, X_test, y_test, k)  
            
            train_accuracies[k].append(train_acc)
            test_accuracies[k].append(test_acc)

    plot_accuracy_graphs(train_accuracies, test_accuracies, k_values)


# In[9]:


def calculate_mean_std_dataset(k_values, train_test_accuracies):
    avg_accuracy = [np.mean(train_test_accuracies[k]) for k in k_values]
    std_accuracy = [np.std(train_test_accuracies[k]) for k in k_values]

    return avg_accuracy, std_accuracy


# In[10]:


def plot_accuracy_graphs(train_accuracies, test_accuracies, k_values):

    mean_training_acc, std_training_acc = calculate_mean_std_dataset(k_values, train_accuracies)
    mean_testing_acc, std_testing_acc = calculate_mean_std_dataset(k_values, test_accuracies)
    
    plt.figure(figsize=(10, 5))
    plt.xlabel('k Value')
    plt.ylabel('Accuracy')
    plt.errorbar(k_values, mean_training_acc, yerr=std_training_acc, label='Training Accuracy', marker='o', capsize=5)
    plt.title('KNN Training Accuracy with Error Bars')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.xlabel('k Value')
    plt.ylabel('Accuracy')
    plt.errorbar(k_values, mean_testing_acc, yerr=std_testing_acc, label='Testing Accuracy', marker='s', capsize=5)
    plt.title('KNN Testing Accuracy with Error Bars')
    plt.legend()
    plt.grid()
    plt.show()


# In[11]:


main()


# In[ ]:





# In[ ]:




