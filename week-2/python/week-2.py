import numpy as np
import os

data_path = os.path.join(os.getcwd(), r'week-2\octave\machine-learning-ex1\ex1', 'ex1data1.txt')
data = np.genfromtxt(data_path, delimiter=',')

X = data[:, 0]
y = data[:, 1]

m = len(y)

