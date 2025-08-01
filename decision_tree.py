#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import random 
import math 
import numpy as np


# In[2]:


def get_dataset(filename):
    dataset = pd.read_csv(filename)
    return dataset


# In[3]:


def calculate_entropy(dataset):
    labels = dataset.iloc[:, -1]
    count_label = labels.value_counts()

    entropy = 0
    for i in count_label:
        probability = i/len(count_label)
        entropy += -probability * math.log2(probability)

    return entropy


# In[4]:


def calculate_information_gain(dataset, feature):

    #information_gain = entropy(parent) - avg(child_entroopy)
    
    dataset_entropy = calculate_entropy(dataset)

    #find unique values for a given attribute in the dataframe
    values = dataset[feature].unique()
    w_entropy = 0
    
    for i in values:
        feature_subset = dataset[dataset[feature] == i]
        feature_prob = len(feature_subset)/len(dataset)
        w_entropy += feature_prob * calculate_entropy(feature_subset)

    information_gain = dataset_entropy - w_entropy

    return information_gain


# In[5]:


def best_feature(dataset, features):
    return max(features, key=lambda feature: calculate_information_gain(dataset, feature))


# In[6]:


def majority_class(dataset):
    labels = dataset.iloc[:, -1]
    return labels.mode()[0]


# In[7]:


class TreeNode():
    def __init__(self, attribute=None, isleaf=False, value=None, label=None, info_gain=None):
        self.attribute = attribute
        self.isleaf = isleaf
        self.label = label
        self.value = value
        self.children = {}


# In[8]:


def build_decision_tree(dataset, features):
    labels = dataset.iloc[:,-1]

    # Stopping criterion 1
    if len(labels.unique()) == 1:
        return TreeNode(isleaf=True, label=labels.iloc[0])

    # Stopping criterion 2
    if not features:
        return TreeNode(isleaf=True, label=majority_class(dataset))

    root_feature = best_feature(dataset, features)
    root = TreeNode(attribute=root_feature)

    values = dataset[root_feature].unique()
    
    for i in values:
        feature_subset = dataset[dataset[root_feature] == i]
        if feature_subset.empty:
            root.children[i] = TreeNode(isleaf = True, label = majority_class(dataset))
        else:
            remaining_attrs = [feature for feature in features if feature != root_feature]
            root.children[i] = build_decision_tree(feature_subset, remaining_attrs)

    return root


# In[9]:


def predict_decision_tree(tree, labels):
    while tree:
        if tree.isleaf:
            return tree.label
        
        test_node_attribute = labels[tree.attribute]
        if test_node_attribute in tree.children:
            tree = tree.children[test_node_attribute]
        else: 
            return None


# In[10]:


def evaluate_model(dataset):
    train_accuracies = []
    test_accuracies = []
    
    for _ in range(100):
        # shuffle the dataset
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        
        training_dataset_size = int(0.8 * len(dataset))
        train_data, test_data = dataset.iloc[:training_dataset_size, :], dataset.iloc[training_dataset_size:, :]
        
        features = dataset.columns[:-1].tolist()
        tree = build_decision_tree(train_data, features)
        
        train_correct = sum(1 for _, row in train_data.iterrows() if predict_decision_tree(tree, row) == row.iloc[-1])
        train_accuracies.append(train_correct / len(train_data))
        
        test_correct = sum(1 for _, row in test_data.iterrows() if predict_decision_tree(tree, row) == row.iloc[-1])
        test_accuracies.append(test_correct / len(test_data))

    return train_accuracies, test_accuracies


# In[11]:


def plot_histogram(accuracies, title):
    plt.hist(accuracies, bins=10, edgecolor="black")
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()


# In[12]:


def main():
    filename = "/Users/saumyapandey/Machine-Learning-Models/datasets/loan.csv" 
    dataset = get_dataset(filename)
    
    training_accuracies, testing_accuracies = evaluate_model(dataset)
    
    mean_training_accuracy = np.mean(training_accuracies)
    std_training_accuracy = np.std(training_accuracies)

    mean_testing_accuracy = np.mean(testing_accuracies)
    std_testing_accuracy = np.std(training_accuracies)
    
    print(f"Training Accuracy: Mean={mean_training_accuracy:.4f}, Std={std_training_accuracy:.4f}")
    print(f"Testing Accuracy: Mean={mean_testing_accuracy:.4f}, Std={std_testing_accuracy:.4f}")
    
    # Plot histograms
    plot_histogram(training_accuracies, "Training Accuracy")
    plot_histogram(testing_accuracies, "Testing Accuracy")


# In[13]:


main()

