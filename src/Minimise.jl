
module Minimise

    using Logging: info
    using JuLIP: dofs, set_dofs!, energy, gradient, hessian, forces, Atoms

    export minimise_newton_method!
    
    # Simple Newton Step Minimise
    # able to also find saddle points
    
    # type and print similar to object and function found in Optim.jl
    mutable struct res_s
        alpha::Float64
        x_tol::Float64
        x_converged::Bool
        g_tol::Float64
        g_converged::Bool
        f_tol::Float64
        f_converged::Bool
        iteration::Int
        iteration_converged::Bool
    end

    """
    print_res_s(r::res_s, g_norm::Float64)
    
    For printing need to call
    `using Logging; Logging.configure(level=INFO)`
    """
    function print_res_s(r::res_s, g_norm::Float64)
        info("Results of Optimization Algorithm")
        info(" * Algorithm: Newton Scheme")
        info(@sprintf(" * Iterations: %i", r.iteration))
        info(@sprintf(" * Convergence: %s", g_norm < r.g_tol))
        info(@sprintf("    * |g(x)|  ≤ %.0e: %s", r.g_tol, g_norm < r.g_tol))
        info(@sprintf("       |g(x)|  = %.2e", g_norm))
        info(@sprintf(" * Reached Maximum Number of Iterations: %s", r.iteration_converged))
    end

    """
    `minimise_newton_method!(atoms::Atoms; gtol::Float64 = 1e-5, max_iterations::Int = 20)`

    Minimise system using Newton scheme.

    ### Arguments
    - `atoms::Atoms` : JuLIP Atoms oject

    #### Optional
    - `g_tol::Float64 = 1e-5` : gradient tolerance, infinity norm
    - `alpha::Float64 = 1.0` : step size
    - `max_iterations::Int = 20` : maximum number of iterations
    - `verbose = 0` : for printing, call  `using Logging; Logging.configure(level=INFO)`
    """
    function minimise_newton_method!(atoms::Atoms; g_tol::Float64 = 1e-5, alpha::Float64 = 1.0, max_iterations::Int = 20, verbose = 0)

        # create similar x_tol and f_tol are not properly done in here only g_tol
        x_tol = 0.0; f_tol = 1e-32 # defaults from Optim
        res = res_s(alpha, x_tol, false, g_tol, false, f_tol, false,  max_iterations, false)

        x = dofs(atoms)
        x0 = copy(x) # original dofs
        iteration = 0
        g_norm = 1
        g_x0_norm = norm(gradient(atoms,x),Inf)
        f_x0_norm = norm(forces(atoms), Inf)
        x_diff_norm = 0.0; f_diff_norm = 0.0

        if verbose > 0 info(@sprintf("    * |g(x0)| = %.2e", g_x0_norm)) end

        while g_norm > g_tol
            x -= alpha*(hessian(atoms, x) \ gradient(atoms, x))

            x_diff_norm = norm(x0 - dofs(atoms), Inf)
            g_norm = norm(gradient(atoms,x), Inf)
            f_diff_norm = norm(f_x0_norm - norm(forces(atoms), Inf), Inf)

            if iteration >= max_iterations
                set_dofs!(atoms, x0) # revert positions back to original given positions
                info("Minimisation Failed.")
                info("Maximum number of iteration reached.")
                pass = 1
                break
            else
                iteration += 1
            end
        end
        set_dofs!(atoms, x)

        res.x_converged = x_diff_norm < x_tol
        res.g_converged = g_norm < g_tol
        res.f_converged = f_diff_norm < f_tol
        res.iteration = iteration
        res.iteration_converged = iteration >= max_iterations

        if verbose > 0
            print_res_s(res, g_norm)
        end

        return res
    end

end # module