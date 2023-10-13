import numpy as np
import pandas
import matplotlib.pyplot as plt

df = pandas.read_excel("Data Take Home Assignment 1 Exercise A.xlsx").loc[22:41]


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


def regression_model(x, o0, o1, o2):
    return (o2 * (x ** 2)) + (o1 * x) + o0


def cost(x, y, o0, o1, o2):
    predictions = regression_model(x, o0, o1, o2)
    error = y - predictions
    cost = (1 / (4 * len(y))) * np.sum(error ** 4)
    return cost


# A.II.2 Linear regression - Data transformation
df = min_max(df.X, df.Y)


def train(x, y, num_iterations):
    # Initialize model parameters and hyperparameters
    o0, o1, o2 = 0.0, 0.0, 0.0  # Initial parameters
    learning_rate = 0.01  # Adjust as needed
    train_size = 1000  # Adjust as needed
    cost_history = []

    x = np.array(x)
    y = np.array(y)

    for iteration in range(num_iterations):
        # Gradient Descent
        for i in range(train_size):
            # Calculate the predictions
            predictions = regression_model(y, o0, o1, o2)

            # Calculate the gradient of the cost function
            gradient_o0 = -(1 / len(x)) * np.sum((x - predictions) ** 3)
            gradient_o1 = -(1 / len(x)) * np.sum((x - predictions) ** 3 * y)
            gradient_o2 = -(1 / len(x)) * np.sum((x - predictions) ** 3 * y ** 2)

            # Update the model parameters
            o0 -= learning_rate * gradient_o0
            o1 -= learning_rate * gradient_o1
            o2 -= learning_rate * gradient_o2

            # Calculate and store the cost for visualization
            current_cost = cost(y, x, o0, o1, o2)
            cost_history.append(current_cost)

        # Visualize the cost over iterations
        plt.plot(range(len(cost_history)), cost_history)  # Use the correct x-values
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title(f'Cost Function over Iterations. Iteration: {iteration + 1}')
        plt.show()

        # Plot the polynomial regression model
        plt.scatter(x, y, label='Data')
        x_values = np.linspace(min(x), max(x), 100)
        y_values = regression_model(x_values, o0, o1, o2)
        plt.plot(x_values, y_values, color='red', label='Polynomial Regression Model')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.title(f'Polynomial Regression Model. Iteration: {iteration + 1}')
        plt.show()

    # Print the final model parameters
    print("Final Model Parameters:")
    print("o0:", o0)
    print("o1:", o1)
    print("o2:", o2)


train(df[0], df[1], 3)
