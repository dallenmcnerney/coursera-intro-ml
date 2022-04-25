import numpy as np
import matplotlib.pyplot as plt


# Shows a scatter plot given x and y arrays
def plot_data(x, y):
    plt.scatter(x, y, c='red', marker='x')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()


# Returns result of sum of squares cost function given x, y, and theta arrays
def compute_cost(x, y, theta):
    m = len(y)
    cost = (1 / (2 * m)) * sum(np.power((np.matmul(x, theta) - y), 2))
    return cost


def gradient_descent(x, y, theta, alpha, iterations=1000):
    m = len(y)
    j_history = np.zeros(iterations)
    for i in range(iterations):
        theta = theta - (alpha * (1/m) * np.matmul(np.transpose(x), (np.matmul(x, theta) - y)))
        j_history[i] = compute_cost(x, y, theta)
    return theta, j_history


def normal_equation(x, y):
    theta = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(x), x)), np.transpose(x)), y)
    return theta


# Load initial data
data = np.genfromtxt('week-2/octave/machine-learning-ex1/ex1/ex1data1.txt', delimiter=',')
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
print('With theta = [0 ; 0]\nCost computed = %.2f\n' % j)
print('Expected cost value (approx) 32.07\n')

# Run secondary test on compute_cost function
j = compute_cost(x_data_ones, y_data, np.array([-1, 2]))
print('With theta = [-1 ; 2]\nCost computed = %.2f\n' % j)
print('Expected cost value (approx) 54.24\n')

# Run gradient descent

theta = np.zeros(2)
theta, j_history = gradient_descent(x_data_ones, y_data, theta, alpha, iterations)
print('Theta found by gradient descent:\n')
print(theta)
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

# Normal equation

theta = normal_equation(x_data_ones, y_data)
print('Theta found by normal equation:\n')
print(theta)
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros([len(theta0_vals), len(theta1_vals)])

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i, j] = compute_cost(x_data_ones, y_data, t)

J_vals = np.transpose(J_vals)

plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 10))
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.plot(theta[0], theta[1], marker='x', markersize=10, color='r')

# Second data set

data = np.genfromtxt('week-2/octave/machine-learning-ex1/ex1/ex1data2.txt', delimiter=',')
x_data = data[:, 0:2]
y_data = data[:, 2]
m = len(y_data)

x_data_ones = np.column_stack((np.ones(m), x_data))

# Calculate theta using the normal equation
theta = normal_equation(x_data_ones, y_data)

# Normal equation results
print('Theta computed from the normal equations: \n')
print(theta)
print('\n')

# Estimate the price of a 1650 sq-ft, 3 br house
price = np.matmul(np.array([1, 1650, 3]), theta)

print('Predicted price of a 1650 sq-ft, 3 br house \n')
print('(using normal equations):\n $%s\n' % price)
