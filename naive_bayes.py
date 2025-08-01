#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import load_training_set, load_test_set
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


# In[2]:


class MultinomialNaiveBayes:
    def __init__(self):
        self.prior_probability = {}
        self.word_probability = {label: defaultdict(float) for label in [0,1]}
        self.vocab = None


# In[3]:


def bag_of_words(dataset, vocab):
    bag = []
    for data in dataset:
        count_of_words = defaultdict(int)
        for word in data:
            if word in vocab:
                count_of_words[word] += 1
        bag.append(count_of_words)
    return bag


# In[4]:


def train_mnb(model, positive_train_bow, negative_train_bow, vocab):
    model.vocab = vocab
    docs_length = len(positive_train_bow) + len(negative_train_bow)
    model.prior_probability[1] = len(positive_train_bow) / docs_length  
    model.prior_probability[0] = len(negative_train_bow) / docs_length

    word_counts = {label: defaultdict(float) for label in [0,1]}
    total_words = {1: 0, 0: 0}

    for data in positive_train_bow:
        for word, count in data.items():
            word_counts[1][word] += count
            total_words[1] += count

    for data in negative_train_bow:
        for word, count in data.items():
            word_counts[0][word] += count
            total_words[0] += count

    vocab_size = len(vocab)

    for label in [0, 1]:
        for word in vocab:
            if word_counts[label][word] == 0:
                model.word_probability[label][word] = 0 
            else:
                model.word_probability[label][word] = word_counts[label][word] / total_words[label]


# In[5]:


def test_mnb(model, dataset_bow):
    scores ={}
    for label in [0,1]:
        scores[label] = model.prior_probability[label]
        
    for word, count in dataset_bow.items():
        if word in model.vocab:
            for label in [0, 1]:
                scores[label] *= model.word_probability[label].get(word, 0) ** count

    return max(scores, key=scores.get)


# In[6]:


def calculate_aprcm(model, test_pos, test_neg):
    
    test_data = []
    for data_pos in test_pos:
        test_data.append((data_pos, 1))
    for data_neg in test_neg:
        test_data.append((data_neg, 0))

    predictions=[]
    for data, _ in test_data:
        predictions.append(test_mnb(model, data))
    
    labels = []
    for _,label in test_data:
        labels.append(label)

    prediction_and_true = list(zip(predictions, labels))
        
    accuracy = np.mean(np.array(predictions) == np.array(labels))

    #true positive
    tp = 0
    for prediction,true in prediction_and_true:
        if prediction == 1 and true == 1:
            tp+=1

    #true negative
    tn = 0
    for prediction,true in prediction_and_true:
        if prediction == 0 and true == 0:
            tn+=1

    #false positive
    fp = 0
    for prediction,true in prediction_and_true:
        if prediction == 1 and true == 0:
            fp+=1

    #false negative
    fn = 0
    for prediction,true in prediction_and_true:
        if prediction == 0 and true == 1:
            fn+=1
    
    confusion_matrix = np.array([[tp, fn], [fp, tn]])
    precision, recall = calculate_precision_recall(tp, fp, fn)

    return accuracy, precision, recall, confusion_matrix


# In[7]:


def calculate_precision_recall(true_positive, false_positive, false_negative):
    precision = 0
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)

    recall = 0
    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)

    return precision, recall


# In[8]:


if __name__ == '__main__':

    (positive_train_set, negative_train_set, vocab) = load_training_set(0.2, 0.2)
    (positive_test_set, negative_test_set) = load_test_set(0.2, 0.2)

    positive_train_bow = bag_of_words(positive_train_set, vocab)
    negative_train_bow = bag_of_words(negative_train_set, vocab)
    positive_test_bow = bag_of_words(positive_test_set, vocab)
    negative_test_bow = bag_of_words(negative_test_set, vocab)

    model = MultinomialNaiveBayes()
    train_mnb(model, positive_train_bow, negative_train_bow, vocab)

    accuracy, precision, recall, confusion_matrix = calculate_aprcm(model, positive_test_bow, negative_test_bow)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix}")

