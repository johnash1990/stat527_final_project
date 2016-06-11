import matplotlib.pyplot as plt
import numpy as np


def center_data(X, cols_to_center):
    """
    Parameters:
    @X {2D numpy array} dataset with variables to be centered
    @cols_to_center {list} list of columns (i.e., variables to center)
    Return:
    @centered_data {2D numpy array} the dataset, but with the specified
    columns centered
    This function centers (i.e., de-means) the data
    """

    # get the count of rows and cols of X
    n_row = X.shape[0]
    n_col = X.shape[1]

    # create a matrix to hold the new, centered data
    centered_data = np.zeros([n_row, n_col])

    # loop over the cols, compute the mean, and subtract it
    for i in range(0, n_col):
        if i in cols_to_center:
            # compute the mean of the original data
            mean = np.mean(X[:, i], axis=0)

            # remove the mean
            centered_data[:, i] = X[:, i] - mean
        else:
            # if the col is not to be centered (e.g., in the case
            # of a categorical variable, leave it as is
            centered_data[:, i] = X[:, i]

    # return the centered data
    return centered_data


def scale_data(X, cols_to_scale):
    """
    Parameters:
    @X {2D numpy array} dataset with variables to be centered
    @cols_to_scale {list} list of columns (i.e., variables) to scale
    such that they have unit variance
    Return:
    @scaled_data {2D numpy array} the dataset, but with the specified
    columns scaled
    This function scales the data to have unit variance
    """
    # get the count of rows and cols
    n_row = X.shape[0]
    n_col = X.shape[1]

    # create a matrix to hold the new, scaled data
    scaled_data = np.zeros([n_row, n_col])

    # loop over the cols, compute the standard deviation, and divide by it
    for i in range(0, n_col):
        if i in cols_to_scale:
            # compute the stdev of the original data
            st_dev = np.std(X[:, i], axis=0)
            scaled_data[:, i] = X[:, i]/st_dev
        else:
            scaled_data[:, i] = X[:, i]

    # return the scaled data
    return scaled_data


def lasso_loss(X, y, beta, lam, shooting=False):
    """
    Parameters:
    @X {2D numpy array} matrix of predictor variables
    @y {numpy array} continuous response variable
    @beta {numpy array} vector of regression coefficients
    @lam {float} regularization parameter
    @shooting {optional, boolean} Flag to indicate if we are
    evaluating the loss for the shooting algorithm which,
    per Murphy (2012) (see paper for full reference), uses
    a lasso loss function without a 0.5 factor in front of the
    RSS(\beta).
    Return:
    @lasso_loss {float} the value of the lasso loss function
    (i.e., a least squares loss plus an L1 penalty) evaluated
    at the given set of inputs
    This function computes the value of the lasso loss function
    at a given set of input data, regression coefficients, and
    regularization parameter value
    """
    
    if not shooting:
        # compute the least squares (LS) loss
        ls_loss = 0.5 * np.linalg.norm(y - np.dot(X, beta))**2
    else:
        # compute the least squares (LS) loss as the RSS
        ls_loss = np.linalg.norm(y - np.dot(X, beta))**2

    # compute the penalty term
    penalty_term = lam * np.sum(np.abs(beta))

    # return the loss function
    return ls_loss + penalty_term


def calc_b_0(orig_y, orig_X, beta):
    """
    Parameters:
    @orig_X {2D numpy array} dataset of predictor variables that has not
    been centered or scaled
    @orig_y {1D numpy array} original response variable that has not been
    centered or scaled
    @beta {1D numpy array} vector of regression coefficients
    Return:
    @b_0 {float} the value of the intercept for the regression problem
    This function recover the value of b_0, i.e., the intercept that was
    lost as a result of centering the data
    """

    # calculate the value of the intercept
    b_0 = np.mean(orig_y) - np.dot(np.mean(orig_X, axis=0), beta)

    # return the intercept
    return b_0


def plot_lasso_coef_profiles(lam_grid, coeffs, col_headers=[]):
    """
    Parameters:
    @lam_grid {1D numpy array} grid regularization parameters used
    @coeffs {2D numpy array} array of regression coefficients where each row
    corresponds to a different lambda value and each column is a different
    predictor
    @col_headers {optional, list} list of column headers to put into a legend
    This function implements the shooting algorithm which uses coordinate
    descent in order to solve the lasso problem. It also makes use of warm
    starts and considers a grid of lambda values with log spacing.
    Return (technically just displayed):
    @plt {matplotlib plot} Plot of coeff vals vs. log_10(lambda)
    """
    # get lambda in log-scale so the graph is more readable
    log_lam = np.log10(lam_grid)

    # loop over the cols of the matrix, where each col contains a different
    # predictor and each row corresponds to the value of that predictor under
    # the given lambdavalue and plot coeff val vs. log lam
    for j in range(0,coeffs.shape[1]):
        plt.plot(log_lam,coeffs[:,j],marker="o")


    # plot formatting
    plt.title("Lasso Coefficient Paths")
    plt.xlabel("Log_10 lambda")
    plt.ylabel("Coefficient Value")

    # check if the col_header list arg is empty, if not, use what
    # the user provided as elements in the legend
    if col_headers:
        plt.legend(col_headers,loc='center left', bbox_to_anchor=(1,0.5),fontsize=14)

    # display the plot
    plt.show()
