import numpy as np
from lasso_utils import *


def warm_start_lasso_shooting_solver_strong_rules(X, y, tol, lam_grid_size, lam_min, lam_max=9999, apply_sequential_strong_rules=True):
    """
    Parameters:
    @X {2D numpy array} matrix of predictor variables
    @y {numpy array} continuous response variable
    @tol {float} converge tolerance (threshold)
    @lam_grid_size {int} number of regularization parameters to try
    @lam_min {float} min val of regularization parameters to try
    @lam_max {optional, float} max val of regularization parameters to try,
    by default we compute the max val s.t. all coefficients in beta=0
    @apply_sequential_strong_rules {optional, boolean} flag whether or not to use
    the sequential strong rules to exclude variables from the optimization
    Return:
    @output {2D numpy array} array where in each row, the index 0 element is
    the lambda value used, the index 1 element is the number of iterations 
    to convergence, and the rest of the elements in the row are coefficient
    estimates in the beta vector.
    This function implements the shooting algorithm which uses coordinate 
    descent in order to solve the lasso problem. It also makes use of warm 
    starts and considers a grid of lambda values with log spacing. This
    function can also implement the sequential strong rules to discard predictors.
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
        
        # make a list of all possible predictor indices to consider in the optimization
        # remove indices for those that are ruled out by the basic strong rules
        vars_to_consider = range(0,n_col)
        
                        
        if apply_sequential_strong_rules and l>=1:
            # compute the vector of quantities to check and compare against the difference in lambda values
            check = np.abs(np.dot(XT,y-np.dot(X,output[l-1,range(2,len(output[0,:]))])))

            # current lam val and that from prev iteration
            lam_l = output[l,0]
            lam_l_1 = output[l-1,0]
            
            # if we should not consider the variable in the optimzation, remove it from
            # the list of indices and set it = 0
            for j in range(0,len(check)):
                if check[j] < (2.*lam_l - lam_l_1):
                    vars_to_consider.remove(j)
                    beta_curr[j] = 0   
        
        # continue to loop and update parameter values until the convergence 
        # threshold is reached
        while converged==False:
            # update the iteration count
            k = k + 1
            
            # get the beta vector from the previous iteration and store it before
            # any of its elements are modified
            beta_old = np.copy(beta_prev)
            
            for j in vars_to_consider:
            #for j in range(0,n_col):
                
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


def lasso_shooting_solver_strong_rules(X, y, lam, tol, verbose=False, apply_basic_strong_rule=False):
    """
    Parameters:
    @X {2D numpy array} matrix of predictor variables
    @y {numpy array} continuous response variable
    @lam {float} regularization parameter
    @tol {float} converge tolerance (threshold)
    @verbose {optional, boolean} flag stating whether or not to print info
    on each iteration to the console
    @apply_basic_strong_rule {optional, boolean} flag whether or not to use
    the basic strong rule to exclude variables from the optimization
    Return:
    @beta_curr {1D numpy array} vector regression coefficients
    This function implements the shooting algorithm which uses coordinate 
    descent in order to solve the lasso problem. It also can apply the basic
    strong rule to remove predictors if desired.
    """
    
    # get the number of cols in X, i.e., the number of predictors
    n_col = X.shape[1]
    
    # initialize the parameter vectors from the previous iteration and the one
    # that results from the current iteration, respectively.
    beta_prev = np.zeros(n_col)
    beta_curr = np.zeros(n_col)
    
    # calc lam_max for use in strong rules
    XT = np.transpose(X)
    lam_max = np.max(np.abs(np.dot(XT,y)))
    
    # make a list of all possible predictor indices to consider in the optimization
    # remove indices for those that are ruled out by the basic strong rules
    vars_to_consider = range(0,n_col)
            
    if apply_basic_strong_rule:
        # calculate the vector of quantities compared to the difference in lambda values
        check = np.abs(np.dot(XT,y))
        
        # loop over all values in the aforementioned vector and compare them to the
        # difference in lambda values, one by one. if the criteria is met, remove the
        # predictor from the set we consider
        for j in range(0,len(check)):
            if check[j] < 2*lam - lam_max:
                # if we remove the variable to consider, its coeff val will remain to 
                # that at which it was initialized, i.e., 0
                vars_to_consider.remove(j)
                if verbose:
                    print "Variable at index %d ruled out by basic strong rule!" % (j)    
    
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

        
        for j in vars_to_consider:
        #for j in range(0,n_col):
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
        
