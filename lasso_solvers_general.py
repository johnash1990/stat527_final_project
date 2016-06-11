import numpy as np
from lasso_utils import *

    
def iterative_soft_thresholding_lasso_solver(X, y, lam, t, tol, verbose=False):
    """
    Parameters:
    @X {2D numpy array} matrix of predictor variables
    @y {numpy array} continuous response variable
    @lam {float} regularization parameter
    @t {float} step size for use in proximal gradient descent
    @tol {float} converge tolerance (threshold)
    @verbose {optional, boolean} flag stating whether or not to print info
    on each iteration to the console
    Return:
    @beta_curr {1D numpy array} vector regression coefficients resulting from 
    ISTA algorithm
    This function implements the iterative soft thresholding algorithm (ISTA) 
    in order to estimate regression coefficients for the lasso problem.
    """
    
    # get the number of cols in X, i.e., the number of predictors
    n_col = X.shape[1]
    
    # transpose X
    XT = np.transpose(X)
    
    # initialize the parameter vectors from the previous iteration and the one
    # that results from the current iteration, respectively.
    beta_prev = np.zeros(n_col)
    beta_curr = np.zeros(n_col)

    # initialize the iteration counter
    k = 0
    
    # set the convergence flag to false
    converged = False
    
    # continue to loop and update parameter values until the convergence 
    # threshold is reached
    while converged == False:
        # take a gradient step
        z_k = beta_prev + np.dot(t*XT, y - np.dot(X, beta_prev))
        
        # update beta by solving the prox with soft-thresholding
        
        # calc (|z_k,j|-t*lam))_+
        pos_part = np.abs(z_k) - t*lam
        
        # only keep the actual positive part
        for i in range(0, n_col):
            if (pos_part[i] <= 0):
                pos_part[i] = 0
        
        # update the current parameter vector       
        beta_curr = np.sign(z_k) * pos_part      

        # compute the value of the lass loss function for the current iteration
        loss_curr = lasso_loss(X, y, beta_curr, lam)

        # check to see if the 2-norm squared for beta_cur-beta_prev
        # is sufficiently small, if so we have converged, if not
        # do another iteration
        two_norm_sq = np.linalg.norm(beta_curr-beta_prev)**2
        
        if (two_norm_sq <= tol):
            converged = True
        
        # if verbose output is desired, print info on the current iteration    
        if verbose:                    
            print "Iter %d done, l2^2=%.03f, Lasso Loss=%.03f" % (k, two_norm_sq, loss_curr)
        
        # overwrite the previous value of the parameter vector    
        beta_prev = np.copy(beta_curr)
        
        # update the iteration count
        k = k + 1

    # once we have converged, return the parameter vector            
    return beta_curr


def fast_iterative_soft_thresholding_lasso_solver(X, y, lam, t, tol, verbose=False):
    """
    Parameters:
    @X {2D numpy array} matrix of predictor variables
    @y {numpy array} continuous response variable
    @lam {float} regularization parameter
    @t {float} step size for use in proximal gradient descent
    @tol {float} converge tolerance (threshold)
    @verbose {optional, boolean} flag stating whether or not to print info
    on each iteration to the console
    Return:
    @beta_curr {1D numpy array} vector regression coefficients resulting from 
    ISTA algorithm
    This function implements the FAST iterative soft thresholding algorithm (FISTA) 
    in order to estimate regression coefficients for the lasso problem. 
    It is quite similar to ISTA, but acceleration is used.
    """
    
    # get the number of cols, which is also the number of predictors
    n_col = X.shape[1]
    
    # transpose X
    XT = np.transpose(X)
        
    
    # initialize the parameter vectors from the previous iteration and one iterationand the one
    # that results from the current iteration, respectively.
    beta_prev_k_1 = np.zeros(n_col)
    beta_prev_k_2 = np.zeros(n_col)
    
    
    # top row is one iterations ago
    # bottom row is two iterations ago
    beta_prev = np.vstack((beta_prev_k_1, beta_prev_k_2))
    
    # initialize the iteration counter
    k = 2
    
    # set the convergence flag to false
    converged = False

    # continue to loop and update parameter values until the convergence 
    # threshold is reached
    while converged==False:
        
        # update theta, convert to float to avoid integer division
        theta_k = beta_prev[0,:] + ((1.0*(k-2))/(1.0*(k+1))) * (beta_prev[0,:] - beta_prev[1,:])
        
        # update z_k, take a gradient step
        z_k = theta_k + np.dot(t*XT, y - np.dot(X,theta_k))
              
        # update beta by solving the prox
        # calc (|z_k,j|-t*lam))_+
        pos_part = np.abs(z_k) - t*lam
        
        # only keep the positive part
        for i in range(0,n_col):
            if (pos_part[i] <= 0):
                pos_part[i] = 0
               
        beta_curr = np.sign(z_k) * pos_part      

        # check to see if the 2-norm squared for beta_cur-beta_prev
        # is sufficiently small, if so we have converged, if not
        # do another iteration
        two_norm = np.linalg.norm(beta_curr-beta_prev[0,:])**2
        
        # compute the value of the lasso loss function for the current iteration
        loss_curr = lasso_loss(X,y,beta_curr,lam)

        # if verbose output is desired, print info on the current iteration
        if verbose:
            print "Iter %d done, l2^2=%.03f, Lasso Loss=%.03f" % (k-1,two_norm,loss_curr)
            
        #beta_prev = np.copy(beta_curr)
        
        # move the top row (one iter ago) to the bottom row (so it is now two iter ago)
        # make the val for one iter ago the current val
        for j in range(0,n_col):
            beta_prev[1,j] = np.copy(beta_prev[0,j])
            beta_prev[0,j] = np.copy(beta_curr[j])
        
        # update the iteration count
        k = k + 1
        
        # check convergence
        if (two_norm <= tol):
            converged = True

    # return the parameter vector            
    return beta_curr


def lasso_shooting(X, y, lam, tol, verbose=False):
    """
    Parameters:
    @X {2D numpy array} matrix of predictor variables
    @y {numpy array} continuous response variable
    @lam {float} regularization parameter
    @tol {float} converge tolerance (threshold)
    @verbose {optional, boolean} flag stating whether or not to print info
    on each iteration to the console
    Return:
    @beta_curr {1D numpy array} vector regression coefficients
    This function implements the shooting algorithm which uses coordinate 
    descent in order to solve the lasso problem.
    """
    
    # get the number of cols in X, i.e., the number of predictors
    n_col = X.shape[1]
    
    # initialize the parameter vectors from the previous iteration and the one
    # that results from the current iteration, respectively.
    beta_prev = np.zeros(n_col)
    beta_curr = np.zeros(n_col)
    
    # initialize the iteration count
    k = 0
    
    # set the convergence flag to false
    converged = False
    
    # continue to loop and update parameter values until the convergence 
    # threshold is reached
    while converged==False:
        # update the iteration count
        k = k + 1
        
        # get the beta vector from the previous iteration and store it before
        # any of its elements are modified
        beta_old = np.copy(beta_prev)
        
        for j in range(0,n_col):
            # del/del_w_j RSS(w) = a_j*w_j - c_j
            a_j = 2 * np.dot(X[:, j],X[:, j])
            
            # proportional to corr between jth variable and residual 
            # excluding the jth variable  
            c_j = 2 * np.dot(X[:, j], (y - np.dot(X, beta_prev) + beta_prev[j]*X[:, j]))
            
            # apply soft thresholding
            pos_part = np.abs(c_j/a_j) - (lam/a_j)
            if pos_part <= 0:
                pos_part = 0
            
            # update the jth coefficient of the parameter vector
            beta_curr[j] = np.sign(c_j/a_j) * pos_part
             
            # VERY IMPORTANT: NEED TO UPDATE BETA_PREV WITH THE NEW jth COEFF
            beta_prev[j] = np.copy(beta_curr[j])
                
        # check to see if the 2-norm squared for beta_cur-beta_prev
        # is sufficiently small, if so we have converged, if not
        # do another iteration
        two_norm = np.linalg.norm(beta_prev-beta_old)**2       
        
        # compute the value of the lasso loss function for the current iteration
        loss_curr = lasso_loss(X, y, beta_curr, lam, True)

        # if verbose output is desired, print info on the current iteration
        if verbose:
            print "Iter %d done, l2^2=%.03f, Loss=%.03f" % (k, two_norm, loss_curr)
        
        # check convergence
        if (two_norm <= tol):
            converged = True     
       
    # return the parameter vector    
    return beta_curr