import torch
import numpy as np


def accuracy(output, target):
    pred = np.argmax(output, axis=1)
    assert pred.shape[0] == len(target)
    correct = np.sum(pred == target)
    return correct / len(target)


def top_k_acc(output, target, k=3):
    pred = np.argsort(output, axis=1)[:, -k:]
    assert pred.shape[0] == len(target)
    correct = 0
    for i in range(k):
        correct += np.sum(pred[:, i] == target)
    return correct / len(target)
