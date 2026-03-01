#!/usr/bin/env python3
"""Comment of Function"""
import numpy as np


def precision(confusion):
    """Precision"""
    classes = confusion.shape[0]
    prec = np.zeros(classes)

    for i in range(classes):
        TP = confusion[i][i]
        FP = np.sum(confusion[:, i]) - TP
        if TP + FP == 0:
            prec[i] = 0
        else:
            prec[i] = TP / (TP + FP)
    return prec
