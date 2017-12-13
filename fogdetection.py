# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 20:20:03 2017

@author: Luqman Ahmad
"""

import numpy as np
from sklearn.svm import SVC

# X : sample value ( 6 sample, 2 fitur )
# y : sample target 
X = np.array([[1,1], [2,2], [3, 3], [10, 10], [12, 12], [13, 13]])
y = np.array([1, 1, 1, 2, 2, 2])

# init classifier as scalable svm
classifier = SVC()
classifier.fit(X, y)

# check prediction. expected class 1 as result ( [ 1.5, 3 ] lebih dekat ke kelas 1)
print(classifier.predict( [[1.5, 3]] ))
print("end")