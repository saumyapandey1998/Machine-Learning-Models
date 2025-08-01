#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import random
import math


# In[2]:


ntree_values = [1, 5, 10, 20, 30, 40, 50]


# In[3]:


def get_dataset(filename):
    dataset = pd.read_csv(filename)
    label = "label" 
    return dataset, label


# In[4]:


class TreeNode:
    def __init__(self, attribute=None, isleaf=False, value=None, label=None, info_gain=None):
        self.attribute = attribute  
        self.isleaf = isleaf        
        self.label = label          
        self.value = value          
        self.children = {}          


# In[5]:


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, min_gain=1e-7):
        self.min_samples_split = min_samples_split # minimal size for split
        self.max_depth = max_depth # max depth
        self.root = None
        self.min_gain = min_gain # minimal gain


# In[6]:


class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=10, min_gain=1e-7):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_gain = min_gain
        self.trees = []


# In[7]:


def calculate_entropy(dataset):
    labels = dataset["label"]
    count_label = labels.value_counts()

    entropy = 0
    for i in count_label:
        probability = i/len(labels)
        entropy += -probability * math.log2(probability)

    return entropy


# In[8]:


def calculate_information_gain(model, dataset, attribute, threshold=None):
    total_entropy = calculate_entropy(dataset)
    weighted_entropy = 0

    if threshold is not None and model.attribute_types[attribute] == "numerical":
        left = dataset[dataset[attribute] <= threshold]
        right = dataset[dataset[attribute] > threshold]

        left_probability = len(left)/len(dataset)
        left_entropy = calculate_entropy(left)

        right_probability = len(right)/len(dataset)
        right_entropy = calculate_entropy(right)

        weighted_entropy = (left_probability * left_entropy) + (right_probability * right_entropy)

    else:
        classes = dataset[attribute].unique()
        weighted_entropy = 0
        total = len(dataset)
        for cls in classes:
            subset = dataset[dataset[attribute] == cls]
            subset_probability = len(subset) / total
            subset_entropy = calculate_entropy(subset)
            weighted_entropy += subset_probability * subset_entropy

    return total_entropy - weighted_entropy


# In[9]:


def get_threshold(values):
    return (values[:-1] + values[1:]) / 2


# In[10]:


def find_best_attribute(model, dataset, attributes):
    best_gain = float('-inf')
    best_attribute = None
    best_threshold = None

    for attribute in attributes:
        if model.attribute_types[attribute] == "numerical":
            values = dataset[attribute].sort_values().unique()
            thresholds = get_threshold(values)
            for threshold in thresholds:
                gain = calculate_information_gain(model, dataset, attribute, threshold)
                if gain > best_gain and gain > model.min_gain:
                    best_gain = gain
                    best_attribute = attribute
                    best_threshold = threshold
        else:
            gain = calculate_information_gain(model, dataset, attribute)
            if gain > best_gain and gain > model.min_gain:
                best_gain = gain
                best_attribute = attribute

    return best_attribute, best_threshold


# In[11]:


def majority_class(data): 
    return data["label"].mode()[0]


# In[12]:


def build_tree(model, dataset, attributes, depth):
    labels = dataset["label"]

    if len(labels.unique()) == 1:
        return TreeNode(isleaf=True, label=labels.iloc[0])

    if not attributes or depth >= model.max_depth:
        return TreeNode(isleaf=True, label= majority_class(dataset))

    best_attribute, best_threshold = find_best_attribute(model, dataset, attributes)
    if best_attribute is None:
        return TreeNode(isleaf=True, label=majority_class(dataset))

    node = TreeNode(attribute=best_attribute, value=best_threshold)
    attribute_type = model.attribute_types[best_attribute]

    if attribute_type == "numerical":
        left = dataset[dataset[best_attribute] <= best_threshold]
        right = dataset[dataset[best_attribute] > best_threshold]
        node.children['<='] = build_tree(model, left, attributes, depth + 1)
        node.children['>'] = build_tree(model, right, attributes, depth + 1)
    else:
        for val in dataset[best_attribute].unique():
            subset = dataset[dataset[best_attribute] == val]
            if subset.empty:
                node.children[val] = TreeNode(isleaf=True, label=majority_class(dataset))
            else:
                new_feats = [f for f in attributes if f != best_attribute]
                node.children[val] = build_tree(model, subset, new_feats, depth + 1)

    return node


# In[13]:


def predict(model, X):
    df = pd.DataFrame(X, columns=model.attribute_names)

    prediction = []

    for _, row in df.iterrows():
        node = model.root
        while not node.isleaf:
            attr = node.attribute
            val = row[attr]
            attribute_type = model.attribute_types[attr]
            if attribute_type == "numerical":
                if val <= node.value:
                    branch = '<=' 
                else:
                    branch = '>'
            else:
                branch = val

            if branch in node.children:
                node = node.children[branch]
            else:
                prediction.append(None)
                break
        else:
            prediction.append(node.label)

    return prediction


# In[14]:


def fit_tree(model, X, y, attribute_types):
    df = pd.DataFrame(X)
    df['label'] = y
    model.attribute_names = list(df.loc[:, df.columns != "label"])
    model.attribute_types = dict(zip(model.attribute_names, attribute_types))
    model.root = build_tree(model, df, model.attribute_names, depth=0)  


# In[15]:


def predict_rf(model, X):
    tree_preds_result = []

    for tree, attributes_idx in model.trees:
        tree_preds = predict(tree, X[:, attributes_idx])
        tree_preds_result.append(tree_preds)

    tree_preds_result = np.array(tree_preds_result)
    tree_preds_result = np.swapaxes(tree_preds_result, 0, 1) 
    return np.array([Counter(row).most_common(1)[0][0] for row in tree_preds_result]) # majority voting 


# In[16]:


def fit_rf(model, X, y, attribute_types):
    model.trees = []
    n_X = X.shape[1]
    m = int(np.sqrt(n_X))
    X = np.array(X)
    y = np.array(y) 

    len_X = len(X)

    for _ in range(model.n_trees):
        b_sample = np.random.choice(len_X, len_X, replace=True) 
        X_sample, y_sample = X[b_sample], y[b_sample]
        attributes_b_sample = pd.Index(range(n_X)).to_series().sample(n=m, replace=False).values

        tree = DecisionTree(min_samples_split=model.min_samples_split,
                            max_depth=model.max_depth,
                            min_gain=model.min_gain)

        sample_attribute_type = [attribute_types[i] for i in attributes_b_sample]
        df_X = pd.DataFrame(X_sample[:, attributes_b_sample], columns=attributes_b_sample)

        fit_tree(tree, df_X, y_sample, sample_attribute_type)
        model.trees.append((tree, attributes_b_sample))


# In[17]:


def stratified_k_fold_split(X, y, k):
    class_labels = np.unique(y)
    build_folds = [[] for _ in range(k)]

    for label in class_labels:
        cls_idx = []
        for i in range (len(y)):
            if y[i] == label:
                cls_idx.append(i)

        np.random.shuffle(np.array(cls_idx))
        split = np.array_split(cls_idx, k)
        for i in range(k):
            build_folds[i].extend(split[i])

    folds = []
    for i in range(k):
        test_idx = np.array(build_folds[i])
        train_idx = []
        for j in range(k):
            if j != i:
                train_idx.extend(build_folds[j])

        train_idx = np.array(train_idx)
        folds.append((np.array(train_idx), test_idx))
    return folds


# In[18]:


def calculate_metrics_for_trees(y_test, y_pred):
    total = len(y_test)
    accuracy = sum(true_val == pred_val for true_val, pred_val in zip(y_test, y_pred)) / total if total != 0 else 0

    y_labels = np.unique(y_test)
    precisions = []
    recalls = []
    f1_score = []

    for label in y_labels:
        tp = sum((true_val == label and pred_val == label) for true_val, pred_val in zip(y_test, y_pred))
        fp = sum((true_val != label and pred_val == label) for true_val, pred_val in zip(y_test, y_pred))
        fn = sum((true_val == label and pred_val != label) for true_val, pred_val in zip(y_test, y_pred))

        if (tp + fp) > 0:
            precision = tp / (tp + fp) 
        else:
            precision = 0

        if (tp + fn) > 0:
            recall = tp / (tp + fn) 
        else:
            recall = 0

        if (precision + recall) != 0:
            f1 = 2 * precision * recall / (precision + recall) 
        else:
            f1 = 0

        precisions.append(precision)
        recalls.append(recall)
        f1_score.append(f1)

    return accuracy, np.mean(precisions), np.mean(recalls), np.mean(f1_score)


# In[19]:


def evaluate_model(model_class, X, y, attribute_types, k=5, n_trees=10):
    folds = stratified_k_fold_split(X, y, k)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for train_idx, test_idx in folds:
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        model = model_class(n_trees=n_trees)
        fit_rf(model, X_train, y_train, attribute_types)
        y_pred = predict_rf(model, X_test)

        test_acc, precision_score, recall_score, f1_score = calculate_metrics_for_trees(y_test, y_pred)
        metrics['accuracy'].append(test_acc)
        metrics['precision'].append(precision_score)
        metrics['recall'].append(recall_score)
        metrics['f1'].append(f1_score)


    results = {}

    for metric in metrics:
        value = metrics[metric]
        results[metric] = np.mean(value)

    return results


# In[20]:


def custom_encoder(dataset):
    u_values = pd.unique(dataset)
    val_to_int = {}

    for ind, val in enumerate(u_values):
        val_to_int[val] = ind

    return series.map(val_to_int)


# In[21]:


def preprocess_dataset(dataset, label):
    dataset = dataset.dropna().reset_index(drop=True)
    X = dataset.loc[:, dataset.columns != "label"]
    y = dataset["label"]

    attribute_types = []

    for col in X.columns:
        if X[col].dtype == 'object':
            attribute_types.append("categorical")
            X.loc[:, col] = X[col].astype(str)
        else:
            attribute_types.append("numerical")

    if y.dtype == 'object':
        y = custom_encoder(y)

    y = y.to_numpy()
    return X.to_numpy(), y, attribute_types, X.columns.tolist()


# In[22]:


def plot_metrics_vs_ntree(X, y, attribute_types, model_class):
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for ntree in ntree_values:
        results = evaluate_model(model_class, X, y, attribute_types, n_trees=ntree)
        for key in metrics:
            metrics[key].append(results[key])
        print(f"n_trees = {ntree}: Accuracy = {results['accuracy']:.4f}, Precision = {results['precision']:.4f}, Recall = {results['recall']:.4f}, F1 = {results['f1']:.4f}")


    for metric in metrics:
        plt.figure()
        plt.plot(ntree_values, metrics[metric], marker='o')
        plt.title(f'{metric.title()} vs. Number of Trees')
        plt.xlabel('Number of Trees')
        plt.ylabel(metric.title())
        plt.grid(True)
        plt.show()


# In[23]:


def main():
    random.seed(42)
    np.random.seed(42)

    dataset, label = get_dataset("/Users/saumyapandey/Machine-Learning-Models/datasets/loan.csv")

    X, y, attribute_types, attribute_names = preprocess_dataset(dataset, label)
    plot_metrics_vs_ntree(X, y, attribute_types, RandomForest)


# In[ ]:


main()

