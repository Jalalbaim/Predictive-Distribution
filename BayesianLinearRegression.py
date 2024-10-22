## Bayesian Linear regression 

import numpy as np
import matplotlib.pyplot as plt


# univariate gaussian data generator
def univariate_gaussian_data_generator(mean, variance):
    #  Irwin–Hall distribution
    sum = 0
    for i in range(12):
        sum += np.random.uniform(0, 1)
    normalized_sum = sum - 6
    std = np.sqrt(variance)
    gaussian = mean + normalized_sum * std
    return gaussian

# polynomial basis linear model data generator
def generate_random_data(n, w, a):
    """
    Inputs:
    n: basis number
    w: weight nx1 vector
    a: variance
    Outputs: a point (x,y)
    """
    x = np.random.uniform(-1, 1)
    X = np.array([x**i for i in range(n)])
    noise = univariate_gaussian_data_generator(0, a)
    y = np.dot(w, X) + noise
    return x, y


# LU decomposition for inverse

def LU_decomposition(matrix):
    n = len(matrix)
    L = np.zeros((n,n))
    U = np.zeros((n,n))
    for i in range(n):
        L[i,i] = 1
        for j in range(i,n):
            U[i,j] = matrix[i,j]
            for k in range(i):
                U[i,j] -= L[i,k] * U[k,j]
        for j in range(i+1,n):
            L[j,i] = matrix[j,i]
            for k in range(i):
                L[j,i] -= L[j,k] * U[k,i]
            L[j,i] /= U[i,i]
    return L,U

def forward_substitution(L,b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i,j] * y[j]
    return y

def backward_substitution(U,y):
    n = len(y)
    x = np.zeros(n)
    for i in range(n-1,-1,-1):
        x[i] = y[i]
        for j in range(i+1,n):
            x[i] -= U[i,j] * x[j]
        x[i] /= U[i,i]
    return x

def inverse(matrix):
    # Inverse of a matrix using LU decomposition
    L,U = LU_decomposition(matrix)
    n = len(matrix)
    inv = np.zeros((n,n))
    for i in range(n):
        b = np.zeros(n)
        b[i] = 1
        y = forward_substitution(L,b)
        x = backward_substitution(U,y)
        inv[:,i] = x
    return inv



def main():
    # inputs 
    n = 4
    a = 1
    b = 10
    w = np.array([1, 2, 3, 4])

    a_inv = 1 / a

    X_matrix = np.zeros((n, 1))
    data_x = []
    data_y = []
    posterior_covariance = np.eye(n) / b
    posterior_mean = np.zeros((n, 1))

    previous_predictive_variance = 0
    i = 0

    while True:
        new_x, new_y = generate_random_data(n, w, a)
        data_x.append(new_x)
        data_y.append(new_y)

        for idx in range(n):  # Changed 'i' to 'idx' to avoid confusion
            X_matrix[idx] = new_x ** idx
        
        # update posterior
        prior_cov_inv = inverse(posterior_covariance)
        X_matrix_T = X_matrix.T

        posterior_covariance = inverse(prior_cov_inv + a_inv * np.dot(X_matrix, X_matrix_T))
        posterior_mean = np.dot(posterior_covariance, a_inv * new_y * X_matrix + np.dot(prior_cov_inv, posterior_mean))

        # print information
        print('Posterior Mean:')
        for value in posterior_mean:
            print(float(value))
        
        print('\nPosterior Variance:')
        for row in posterior_covariance:
            print("  ".join(f"{col:.5f}" for col in row))

        # predictive distribution
        predictive_mean = float(np.dot(X_matrix_T, posterior_mean))
        predictive_variance = 1 / a + float(np.dot(np.dot(X_matrix_T, posterior_covariance), X_matrix))

        print('\nPredictive Distribution ~ N(', predictive_mean, ',', predictive_variance, ')')

        # check for convergence only after a minimum number of iterations
        if abs(predictive_variance - previous_predictive_variance) < 1e-6 and i >= 50:
            break

        previous_predictive_variance = predictive_variance
        i += 1

        if i == 10:
            data_x_plt1 = np.copy(data_x)
            data_y_plt1 = np.copy(data_y)
            mean_plt1 = np.copy(posterior_mean)
            cov_plt1 = np.copy(posterior_covariance)
        elif i == 50:
            data_x_plt2 = np.copy(data_x)
            data_y_plt2 = np.copy(data_y)
            mean_plt2 = np.copy(posterior_mean)
            cov_plt2 = np.copy(posterior_covariance)

    # plot
    plt.figure(1)
    # plot ground truth
    plt.subplot(221)
    plt.title('Ground Truth')
    plt.xlim(-2.0, 2.0)
    plt.ylim(-20.0, 20.0)
    x_values = np.linspace(-2.0, 2.0, 30)
    y_values = np.zeros(30)
    y_upper = np.zeros(30)
    y_lower = np.zeros(30)

    for i in range(30):
        y_values[i] = sum(w[n] * (x_values[i] ** n) for n in range(n))
        y_upper[i] = y_values[i] + a
        y_lower[i] = y_values[i] - a

    plt.plot(x_values, y_values, 'k', linewidth=1)
    plt.plot(x_values, y_upper, 'r', linewidth=1)
    plt.plot(x_values, y_lower, 'r', linewidth=1)

    # plot posterior 10 
    plt.subplot(222)
    plt.title('After 10 data points')
    plt.xlim(-2.0, 2.0)
    plt.ylim(-20.0, 20.0)
    
    x_values = np.linspace(-2.0, 2.0, 30)
    y_values = np.zeros(30)
    y_upper = np.zeros(30)
    y_lower = np.zeros(30)

    for i in range(30):
        design_matrix = np.array([x_values[i] ** j for j in range(n)])
        predictive_variance = 1 / b + float(np.dot(design_matrix, np.dot(cov_plt1, design_matrix.T)))
        y_values[i] = float(np.dot(design_matrix, mean_plt1).item())
        y_upper[i] = y_values[i] + predictive_variance
        y_lower[i] = y_values[i] - predictive_variance

    plt.plot(x_values, y_values, 'k', linewidth=1)
    plt.plot(x_values, y_upper, 'r', linewidth=1)
    plt.plot(x_values, y_lower, 'r', linewidth=1)
    plt.scatter(data_x_plt1, data_y_plt1, color='b', s=10)

    # plot posterior 50
    plt.subplot(223)
    plt.title('After 50 data points')
    plt.xlim(-2.0, 2.0)
    plt.ylim(-20.0, 20.0)
    
    x_values = np.linspace(-2.0, 2.0, 30)
    y_values = np.zeros(30)
    y_upper = np.zeros(30)
    y_lower = np.zeros(30)

    for i in range(30):
        design_matrix = np.array([x_values[i] ** j for j in range(n)])
        predictive_variance = 1 / b + float(np.dot(design_matrix, np.dot(cov_plt2, design_matrix.T)))
        y_values[i] = float(np.dot(design_matrix, mean_plt2).item())
        y_upper[i] = y_values[i] + predictive_variance
        y_lower[i] = y_values[i] - predictive_variance

    plt.plot(x_values, y_values, 'k', linewidth=1)
    plt.plot(x_values, y_upper, 'r', linewidth=1)
    plt.plot(x_values, y_lower, 'r', linewidth=1)
    plt.scatter(data_x_plt2, data_y_plt2, color='b', s=10)

    # Final result
    plt.subplot(224)
    plt.title('Final Result')
    plt.xlim(-2.0, 2.0)
    plt.ylim(-20.0, 20.0)
    
    x_values = np.linspace(-2.0, 2.0, 30)
    y_values = np.zeros(30)
    y_upper = np.zeros(30)
    y_lower = np.zeros(30)

    for i in range(30):
        design_matrix = np.array([x_values[i] ** j for j in range(n)])
        predictive_variance = 1 / b + float(np.dot(design_matrix, np.dot(posterior_covariance, design_matrix.T)))
        y_values[i] = float(np.dot(design_matrix, posterior_mean).item())
        y_upper[i] = y_values[i] + predictive_variance
        y_lower[i] = y_values[i] - predictive_variance

    plt.plot(x_values, y_values, 'g', linewidth=1)
    plt.plot(x_values, y_upper, 'r', linewidth=1)
    plt.plot(x_values, y_lower, 'r', linewidth=1)
    plt.scatter(data_x, data_y, color='b', s=10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    
