import numpy as np
import os
import matplotlib.pyplot as plt

data_path = os.path.join(os.getcwd(), r'week-2\octave\machine-learning-ex1\ex1', 'ex1data1.txt')


def plot_data(x, y):
    plt.scatter(x, y, c='red', marker='x')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()


def compute_cost(x, y, theta):
    m = len(y)
    cost = (1 / (2 * m)) * sum(np.power((np.matmul(x, theta) - y), 2))
    return cost


# Load initial data
data = np.genfromtxt(data_path, delimiter=',')
x_data = data[:, 0]
y_data = data[:, 1]
m = len(y_data)

# Generate scatter plot of data
plot_data(x_data, y_data)


x_data_ones = np.column_stack((np.ones(m), x_data))
theta = np.zeros(2)
iterations = 1500
alpha = 0.01

# Test the compute_cost function
j = compute_cost(x_data_ones, y_data, theta)

# Run secondary test on compute_cost function
j = compute_cost(x_data_ones, y_data, np.array([-1, 2]))