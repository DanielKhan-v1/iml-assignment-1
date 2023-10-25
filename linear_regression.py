import numpy as np
import pandas
import matplotlib.pyplot as plt

# Function for Min-Max scaling
def min_max(X, Y):
    # Define the range for scaling (e.g., [0, 1])
    min_range = 0
    max_range = 1

    # Calculate the min and max values for each column
    min_X = min(X)
    max_X = max(X)
    min_Y = min(Y)
    max_Y = max(Y)

    # Perform Min-Max scaling
    scaled_X = [(x - min_X) / (max_X - min_X) * (max_range - min_range) + min_range for x in X]
    scaled_Y = [(y - min_Y) / (max_Y - min_Y) * (max_range - min_range) + min_range for y in Y]

    return scaled_X, scaled_Y

# Function for performing Linear Regression using Least Squares
def least_squares(x, y):
    # Ensure that x and y are NumPy arrays
    x = np.array(x)
    y = np.array(y)

    # Calculate the number of data points
    n = len(x)

    # Calculate the necessary sums
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x_squared = np.sum(x ** 2)
    sum_xy = np.sum(x * y)

    # Calculate the regression coefficients (slope and intercept)
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / n

    # Print the regression parameters
    print(f"Regression Equation: y = {slope:.2f}x + {intercept:.2f}")

    # Create the regression line
    regression_line = slope * x + intercept

    # Plot the data points and regression line
    plt.scatter(x, y, label='Data Points')
    plt.plot(x, regression_line, color='red', label='Regression Line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Linear Regression using Least Squares')
    plt.grid(True)
    plt.show()

# Function to calculate the Mean Squared Error (MSE)
def cost(y_true, y_pred):
    n = len(y_true)
    squared_errors = [(y_true[i] - y_pred[i]) ** 2 for i in range(n)]
    resulting_cost = (1 / (2 * n)) * sum(squared_errors)
    return resulting_cost

# Function to train a linear regression model using Gradient Descent
def train(x, y, num_iterations):
    x = np.array(x)
    y = np.array(y)

    # Randomly initialize the regression parameters (slope and intercept)
    np.random.seed(0)  # For reproducibility
    initial_slope = np.random.rand()
    initial_intercept = np.random.rand()
    learning_rate = 0.01  # Choose an appropriate learning rate

    # Create an empty list to store costs for each iteration
    costs = []

    for iteration in range(num_iterations):
        print("-------------------------------------")
        print("Iteration number: " + str(iteration + 1))
        # Perform the first iteration of gradient descent
        train_size = 500
        for i in range(train_size):
            # Calculate the predicted values
            predicted = initial_slope * x + initial_intercept

            # Calculate the cost using the MSE function
            current_cost = cost(y, predicted)

            # Calculate the gradient of the cost function
            gradient_slope = (-1 / len(x)) * np.sum(x * (y - predicted))
            gradient_intercept = (-1 / len(x)) * np.sum(y - predicted)

            # Update the regression parameters
            initial_slope -= learning_rate * gradient_slope
            initial_intercept -= learning_rate * gradient_intercept

        # Append the cost for this iteration to the list
        costs.append(current_cost)

        # Print the chosen random initial parameters, the computed values, and the cost
        print(f"Initial Slope: {initial_slope:.2f}")
        print(f"Initial Intercept: {initial_intercept:.2f}")
        print(f"Cost after iteration {iteration + 1}: {current_cost:.2f}")
        print("-------------------------------------")

        # Plot the data points and the regression line
        plt.scatter(x, y, label='Data Points')
        plt.plot(x, initial_slope * x + initial_intercept, color='red',
                 label=f'Regression Line (Iteration {str(iteration + 1)})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.title(f'Linear Regression with Gradient Descent - Iteration {str(iteration + 1)}')
        plt.grid(True)
        plt.show()

    # Plot the cost over iterations
    plt.plot(range(1, num_iterations + 1), costs)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost vs. Iteration')
    plt.grid(True)
    plt.show()
# A.I.1 Linear regression - Data acquisition
df = pandas.read_excel("Data Take Home Assignment 1 Exercise A.xlsx").loc[20:39]
print(df)

# A.I.2 Linear regression - Data transformation
df = min_max(df.X, df.Y)


# A.I.3 Linear regression - Least Squares
least_squares(df[0], df[1])

# A.I.4 Linear regression - Gradient Descent training
train(df[0], df[1], 5)
