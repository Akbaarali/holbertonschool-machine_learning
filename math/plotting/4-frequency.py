#!/usr/bin/env python3

"""A module that does the trick"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Not that fast"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.hist(student_grades, bins=np.arange(0, 101, 10), ec="black")
    plt.title("Project A")
    plt.ylim(0, 30)
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 101, 10))
    plt.show()
