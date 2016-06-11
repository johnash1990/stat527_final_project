import numpy as np
from lasso_utils import *

def warm_start_ISTA_lasso_solver(X, y, t, tol, lam_grid_size, lam_min, lam_max=9999):
    """
    Parameters:
    @X {2D numpy array} matrix of predictor variables
    @y {numpy array} continuous response variable
    @t {float} step size for use in proximal gradient descent
    @tol {float} converge tolerance (threshold)
    @lam_grid_size {int} number of regularization parameters to try
    @lam_min {float} min val of regularization parameters to try
    @lam_max {optional, float} max val of regularization parameters to try,
    by default we compute the max val s.t. all coefficients in beta=0
    Return:
    @output {2D numpy array} array where in each row, the index 0 element is
    the lambda value used, the index 1 element is the number of iterations 
    to convergence, and the rest of the elements in the row are coefficient
    estimates in the beta vector.
    This function implements the iterative soft thresholding algorithm (ISTA) 
    in order to estimate regression coefficients for the lasso problem. It also
    makes use of warm starts and considers a grid of lambda values with log spacing.
    """
    
    # get the number of cols in X, i.e., the number of predictors
    n_col = X.shape[1]
    
    # transpose X
    XT = np.transpose(X)
    
    # if the default arg is chosen, use lam_max, otherwise use user-specified max val
    if lam_max == 9999:
        # take infinity norm
        lam_max = np.max(np.abs(np.dot(XT,y)))
    
    # make a grid of lam with specified number of points and log spacing
    lam_grid = 10.**np.linspace(np.log10(lam_max), np.log10(lam_min), lam_grid_size)
    
    # initialize the array to store the output
    # col0=lam, col1=#iter, col2-coln=beta coeffs
    output = np.zeros([len(lam_grid),n_col+2])
    
    # fill the 1st col with lambdas
    for i in range(0,len(lam_grid)):
        output[i,0] = lam_grid[i]

    # initialize the parameter vectors from the previous iteration and the one
    # that results from the current iteration, respectively.
    beta_curr = np.zeros(n_col)
    
    # loop over all values of the regularization parameter in the grid    
    for j in range(0,len(lam_grid)):

        # set the convergence flag to false
        converged = False
        
        # initialize beta_prev to whatever the last solution was
        # this is the WARM START PART
        beta_prev = np.copy(beta_curr)
        
        # initialize the iteration counter
        k = 0
        
        # continue to loop and update parameter values until the convergence 
        # threshold is reached
        while converged==False:
            # take a gradient step
            z_k = beta_prev + np.dot(t*XT, y - np.dot(X, beta_prev))
                        
            # update beta by solving the prox
            # calc (|z_k,j|-t*lam))_+
            pos_part = np.abs(z_k) - t*lam_grid[j]
            
            for i in range(0,n_col):
                if (pos_part[i] <= 0):
                    pos_part[i] = 0
            
            # update the current parameter vector        
            beta_curr = np.sign(z_k) * pos_part      
                
            # check to see if the 2-norm squared for beta_cur-beta_prev
            # is sufficiently small, if so we have converged, if not
            # do another iteration
            two_norm = np.linalg.norm(beta_curr-beta_prev)**2
            
            if (two_norm <= tol):
                converged = True
    
            # overwrite the previous value of the parameter vector        
            beta_prev = np.copy(beta_curr)
            
            # update the iteration count
            k = k + 1
            
        # insert the param vector and # of iter into the output matrix          
        output[j, 1] = k
        for i in range(0, n_col):
            output[j, i+2] = beta_curr[i]

    return output


def warm_start_FISTA_lasso_solver(X, y, t, tol, lam_grid_size, lam_min, lam_max=9999):
    """
    Parameters:
    @X {2D numpy array} matrix of predictor variables
    @y {numpy array} continuous response variable
    @t {float} step size for use in proximal gradient descent
    @tol {float} converge tolerance (threshold)
    @lam_grid_size {int} number of regularization parameters to try
    @lam_min {float} min val of regularization parameters to try
    @lam_max {optional, float} max val of regularization parameters to try,
    by default we compute the max val s.t. all coefficients in beta=0
    Return:
    @output {2D numpy array} array where in each row, the index 0 element is
    the lambda value used, the index 1 element is the number of iterations 
    to convergence, and the rest of the elements in the row are coefficient
    estimates in the beta vector.
    This function implements the fast iterative soft thresholding algorithm (FISTA) 
    in order to estimate regression coefficients for the lasso problem; the fast part
    means that we make use of acceleration in the proximal gradient descent problem.
    It also makes use of warm starts and considers a grid of lambda values with log spacing.
    """
    
    # get the number of cols in X, i.e., the number of predictors
    n_col = X.shape[1]
    
    # transpose X
    XT = np.transpose(X)  
    
    # if the default arg is chosen, use lam_max, otherwise use user-specified max val
    if lam_max == 9999:
        # take infinity norm
        lam_max = np.max(np.abs(np.dot(XT,y)))
    
    # make a grid of lam with specified number of points and log spacing
    lam_grid = 10.**np.linspace(np.log10(lam_max), np.log10(lam_min), lam_grid_size)
        
    # initialize the array to store the output
    # col0=lam, col1=#iter, col2-coln=beta coeffs
    output = np.zeros([len(lam_grid),n_col+2])
    
    # fill the 1st col with lambdas
    for i in range(0,len(lam_grid)):
        output[i,0] = lam_grid[i]
    
    # initialize beta_curr
    beta_curr = np.zeros(n_col)
    
    # loop over all values of the regularization parameter in the grid    
    for j in range(0,len(lam_grid)):

        # set the convergence flag to false
        converged = False
        
        # initialize beta_prev to whatever the last solution was
        # this is the WARM START part
        beta_prev = np.vstack((np.copy(beta_curr), np.copy(beta_curr)))
        
        # initialize the iteration counter
        k = 2
        
        # continue to loop and update parameter values until the convergence 
        # threshold is reached
        while converged==False:
            # update theta, convert to float to avoid integer division
            theta_k = beta_prev[0,:] + ((1.0*(k-2))/(1.0*(k+1))) * (beta_prev[0,:] - beta_prev[1,:])
            
            # update z_k, take a gradient step
            z_k = theta_k + np.dot(t*XT, y - np.dot(X,theta_k))
                  
            # update beta by solving the prox
            # calc (|z_k,j|-t*lam))_+
            pos_part = np.abs(z_k) - t*lam_grid[j]
            
            # only keep the positive part
            for i in range(0,n_col):
                if (pos_part[i] <= 0):
                    pos_part[i] = 0
                   
            beta_curr = np.sign(z_k) * pos_part      
    
            # check to see if the 2-norm squared for beta_cur-beta_prev
            # is sufficiently small, if so we have converged, if not
            # do another iteration
            two_norm = np.linalg.norm(beta_curr-beta_prev[0,:])**2
        
            # move the top row (one iter ago) to the bottom row (so it is now two iter ago)
            # make the val for one iter ago the current val
            for l in range(0,n_col):
                beta_prev[1,l] = np.copy(beta_prev[0,l])
                beta_prev[0,l] = np.copy(beta_curr[l])
            
            # update the iteration count
            k = k + 1
        
            # check convergence
            if (two_norm <= tol):
                converged = True
        
        
        # insert the param vector and # of iter into the output matrix          
        output[j, 1] = k
        for i in range(0, n_col):
            output[j, i+2] = beta_curr[i]

    return output


def warm_start_lasso_shooting_solver(X, y, tol, lam_grid_size, lam_min, lam_max=9999):
    """
    Parameters:
    @X {2D numpy array} matrix of predictor variables
    @y {numpy array} continuous response variable
    @tol {float} converge tolerance (threshold)
    @lam_grid_size {int} number of regularization parameters to try
    @lam_min {float} min val of regularization parameters to try
    @lam_max {optional, float} max val of regularization parameters to try,
    by default we compute the max val s.t. all coefficients in beta=0
    Return:
    @output {2D numpy array} array where in each row, the index 0 element is
    the lambda value used, the index 1 element is the number of iterations 
    to convergence, and the rest of the elements in the row are coefficient
    estimates in the beta vector.
    This function implements the shooting algorithm which uses coordinate 
    descent in order to solve the lasso problem. It also makes use of warm 
    starts and considers a grid of lambda values with log spacing.
    """
    # get the number of cols in X, i.e., the number of predictors
    n_col = X.shape[1]
    
    # transpose X
    XT = np.transpose(X)  
    
    # if the default arg is chosen, use lam_max, otherwise use user-specified max val
    if lam_max == 9999:
        # take infinity norm
        lam_max = 2 * np.max(np.abs(np.dot(XT,y)))
    
    # make a grid of lam with specified number of points and log spacing
    lam_grid = 10.**np.linspace(np.log10(lam_max), np.log10(lam_min), lam_grid_size)
        
    # initialize the array to store the output
    # col0=lam, col1=#iter, col2-coln=beta coeffs
    output = np.zeros([len(lam_grid), n_col+2])
    
    # fill the 1st col with lambdas
    for i in range(0,len(lam_grid)):
        output[i,0] = lam_grid[i]

    # initialize beta_curr
    beta_curr = np.zeros(n_col)
    
    # loop over all values of the regularization parameter in the grid    
    # use "l" as the index since we already have an inner for loop on j
    for l in range(0,len(lam_grid)):
       
        # initialize the iteration count
        k = 0
        
        # set the convergence flag to false
        converged = False
        
        # initialize beta_prev to whatever the last solution was
        # this is the WARM START PART
        beta_prev = np.copy(beta_curr)
        
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
                pos_part = np.abs(c_j/a_j) - (lam_grid[l]/a_j)
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
           
            # check convergence
            if (two_norm <= tol):
                converged = True       
        
        # insert the param vector and # of iter into the output matrix          
        output[l, 1] = k
        for i in range(0, n_col):
            output[l, i+2] = beta_curr[i]

    return output   
