def secant_method(F, s0, s1, h=0.01, tol=1e-10, max_iter=100):
    # Notation
    '''
    s_past = s_{n-1} [initial guess s0]
    s_current = s_n [initial guess s1]
    f_past = F(s_{n-1}) = F(s_past)
    f_current = F(s_n) = F(s_current)
    s_next = s_{n+1} [uses secant formula]
    '''
    # Keep track of all s values for table
    s_values = [s0, s1]
    
    s_past, s_current = s0, s1
    F_past = F(s_past, h=h)
    F_current = F(s_current, h=h)
    
    # Keep track of F(s) values 
    F_values = [F_past, F_current]

    for _ in range(max_iter):
        # Use secant formula to find the next time step
        s_next = s_current - F_current * (s_current - s_past) / (F_current - F_past)
        
        F_next = F(s_next, h=h)
        
        s_values.append(s_next)
        F_values.append(F_next)

        # Check for convergence
        if abs(F_next) < tol and abs(s_next - s_current) < tol:
            return s_values, F_values

        # Update for the next iteration
        s_past, s_current = s_current, s_next
        F_past, F_current = F_current, F_next
        

    # If we reach here, it means we did not converge within the maximum iterations
    print("Warning: Maximum iterations reached without convergence.")
    return s_values, F_values