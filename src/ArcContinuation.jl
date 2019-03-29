# Pseudo Arc-Length Continuation Scheme

module ArcContinuation

    using JuLIP: JVec, JVecsF, AbstractAtoms, cutoff, neighbourlist, FixedCell,
                        mat, calculator, zerovecs, AbstractConstraint, hessian, dofs,
                        Atoms, set_positions!, constraint, calculator, set_dofs!,
                        get_positions, forces
    using JuLIP.Potentials: PairPotential, npairs, hess

    include("Minimise.jl") # need res_s and print_res_s
    # would be nice to get rid of this or match it better with Optim

    export hessian_k, forces_k, gradient_k

    #=
    - the crucial bit is to be able to differentiate the gradient of the energy
    with respect to k
    - k is a scalar, so the derivative of the gradient with respect to k will
    also be a vector of the same length as the gradient
    =#

    # JuLIP implentation as of 28.03.19
    #=
    function hess(V::PairPotential, r, R)
        R̂ = R/r
        P = R̂ * R̂'
        dV = (@D V(r))/r
        return ((@DD V(r)) - dV) * P + dV * one(JMatF)
     end

    grad(V::PairPotential, r::Real, R::JVec) = (evaluate_d(V, r) / r) * R

    function forces(V::PairPotential, at::AbstractAtoms)
        nlist = neighbourlist(at, cutoff(V))::PairList
        dE = zerovecs(length(at))
        @simd for n = 1:npairs(nlist)
            @inbounds dE[nlist.i[n]] += grad(V, nlist.r[n], nlist.R[n])
        end
        return dE
    end
     =#

    # derivative of the gradient with respect to k is and each contribution is
    """
    `hessian_k(V::PairPotential, r::Real, R::JVec, i, j, u::JVecsF)`

    derivative of the gradient with respect to k is and each contribution is

    ### Arguments
    - `V::PairPotential`
    - `r::Real`
    - `R::JVec`
    - `i`
    - `j`
    - `u::JVecsF` : inital displacements such that k is independent of u, k*u
    """
    function hessian_k(V::PairPotential, r::Real, R::JVec, i, j, u::JVecsF)
        hess(V,r,R)*(u[i] - u[j])
    end

    """
    `forces_k(atoms::AbstractAtoms, u::JVecsF)`

    `forces_k(V::PairPotential, atoms::AbstractAtoms, u::JVecsF)`

    adapted forces, which was used by the original gradient code

    ### Arguments
    - `V::PairPotential`
    - `atoms::AbstractAtoms` : Atoms object
    - `u::JVecsF` : inital displacements such that k is independent of u, k*u
    """
    function forces_k(V::PairPotential, atoms::AbstractAtoms, u::JVecsF)
        nlist = neighbourlist(atoms, cutoff(V))
        dE = zerovecs(length(atoms))
        @simd for n = 1:npairs(nlist)
            @inbounds dE[nlist.i[n]] += hessian_k(V, nlist.r[n], nlist.R[n],nlist.i[n],nlist.j[n], u)
        end
        return dE
    end
    forces_k(atoms::AbstractAtoms, u::JVecsF) = forces_k(calculator(atoms), atoms, u)

    """
    `gradient_k(atoms::AbstractAtoms, u::JVecsF)`

    `gradient_k(atoms::AbstractAtoms, cons::AbstractConstraint, u::JVecsF)`

    derivative of the gradient with respect to k

    ### Arguments
    - `atoms::AbstractAtoms` : Atoms object
    - `cons::AbstractConstraint` : constraint
    - `u::JVecsF` : inital displacements such that k is independent of u, k*u
    """
    gradient_k(atoms::AbstractAtoms, cons::AbstractConstraint, u::JVecsF) =
            scale!(mat(forces_k(atoms, u))[cons.ifree], 1.0)
    gradient_k(atoms::AbstractAtoms, u::JVecsF) = 
                        gradient_k(atoms, constraint(atoms), u)

    """
    `hessian_arc(atoms::AbstractAtoms, u::JVecsF, xd::Array{Float64})`

    To solve the extended system, need to assemble it's hessian.
    The system is now of size (3N+1) so the hessian is of size (3N+1)x(3N+1).
    The 3Nx3N part is, as before, of the `hessian(atoms)`.
    The extra column is the `gradient_k(atoms, u)`.
    The extra row is the xdot, `xd`.

    ### Arguments
    - `atoms::AbstractAtoms` : Atoms object
    - `u::JVecsF` : inital displacements such that k is independent of u, k*u
    - `xd=Array{Float64}` : xdot() as below
    """
    function hessian_arc(atoms::AbstractAtoms, u::JVecsF, xd::Array{Float64})
        ha = hessian(atoms)
        Ia = findnz(ha)[1]
        Ja = findnz(ha)[2]
        Va = findnz(ha)[3]
        N = length(dofs(atoms)) + 1
        I = [ Ia; 1:(N-1); N*ones(Int64,N) ]
        J = [ Ja; N*ones(Int64,N-1); 1:N ]
        V = [ Va; gradient_k(atoms, u); xd ]
        h = sparse(I, J, V)

        return h
    end

    """
    `xdot(atoms::AbstractAtoms, u::JVecsF)`

    `xdot(atoms::AbstractAtoms, u::JVecsF, xdot0::Array{Float64})`

    For the 'pseudo arc-length continuation scheme' it is crucial to finds 
    tangents.

    `xdot(atoms::AbstractAtoms, u::JVecsF)` is expensive and only works away from the 
    bifurcation point.

    `xdot(atoms::AbstractAtoms, u::JVecsF, xdot0::Array{Float64})` is a simpler
    that works atoms the bifurcation points, but it requires a previous guess.
    So use the first as the initial step and then use the second.

    ### Arguments
    - `atoms::AbstractAtoms` : Atoms object
    - `u::JVecsF` : inital displacements such that k is independent of u, k*u

    Extra
    - `xdot0::Array{Float64}` : guess of xdot
    """
    function xdot(atoms::AbstractAtoms, u::JVecsF)
        gk = gradient_k(atoms, u)
        f1 = -hessian(atoms) \ gk
        kdot = 1.0/(sqrt(dot(f1,f1) + 1))
        udot = kdot*(f1)
        xdot = [udot; kdot]
        return xdot
    end
    function xdot(atoms::AbstractAtoms, u::JVecsF, xdot0::Array{Float64})
        N = length(xdot0)
        RHS = zeros(N)
        RHS[N] = 1.0
        xdot = hessian_arc(atoms, u, xdot0) \ RHS
        return xdot
    end

    """
    `minimise_newton_method_arc!(atoms::Atoms, pos_cryst::JVecsF, u_start::JVecsF,
                                k::Float64, xd::Array{Float64}; 
                                ds::Float64 = 0.001, g_tol::Float64 = 1e-5, 
                                alpha::Float64 = 1.0, max_iterations::Int = 20, verbose = 0)`

    Arc length continuation scheme minimisations using Newton scheme

    ### Arguments
    - `atoms::Atoms` : JuLIP Atoms oject
    - `pos_cryst::JVecsF` : crystal positions
    - `u_start::JVecsF` : the initial displacements
    - `k::Float64` : stress intensity factor
    - `xd::Array{Float64}` : tangent at the previous solution, xdot

    #### Optional
    - `ds::Float64 = 0.001` : step size
    - `g_tol::Float64 = 1e-5` : gradient tolerance, infinity norm
    - `alpha::Float64 = 1.0` : step size
    - `max_iterations::Int = 20` : maximum number of iterations
    - `verbose = 0`
    """
    function minimise_newton_method_arc!(atoms::Atoms, pos_cryst::JVecsF, u_start::JVecsF,
                                k::Float64, xd::Array{Float64}; 
                                ds::Float64 = 0.001, g_tol::Float64 = 1e-5, 
                                alpha::Float64 = 1.0, max_iterations::Int = 20, verbose = 0)
    
        # create similar x_tol and f_tol are not properly done in here only g_tol
        x_tol = 0.0; f_tol = 1e-32 # defaults from Optim
        res = Minimise.res_s(alpha, x_tol, false, g_tol, false, f_tol, false,  max_iterations, false)

        x = dofs(atoms)
        N = length(x)

        # a crude way of extracting the atomistic correction displacements:
        x_help = copy(dofs(atoms))
        set_positions!(atoms,pos_cryst+k*u_start)
        # we assemble the extended system
        x_ext = [x_help-dofs(atoms);k]
        #now we have them, we revert back to the configuration we started with
        set_dofs!(atoms,x_help)

        x0 = copy(x) # original dofs
        x_ext0 = copy(x_ext) #original extended system

        #we replace original configuration and original k with new guess:
        lala = copy(get_positions(atoms))
        u0_old = k*u_start
        k = k +ds*xd[N+1]
        u0_new = k*u_start
        set_positions!(atoms,lala-u0_old+u0_new)
        x = dofs(atoms)
        set_dofs!(atoms,x + ds*xd[1:N])

        #again the crude way of extracting the atomistic information:
        x = dofs(atoms)
        x_help = copy(dofs(atoms))
        set_positions!(atoms,pos_cryst+k*u_start)
        x_ext = [x_help-dofs(atoms);k]
        set_dofs!(atoms,x_help)

        iteration = 0
        g_ext_norm = 1

        #this is the extra equation that closes the system:
        extra_eqn = dot(x_ext-x_ext0,xd) - ds
        g_ext = [gradient(atoms);extra_eqn]
        g_ext0_norm = norm(g_ext,Inf)

        f_x0_norm = norm(forces(atoms), Inf)
        x_ext_diff_norm = 0.0; f_diff_norm = 0.0
        if verbose > 0 info(@sprintf("    * |g(x0)| = %.2e", g_ext0_norm)) end
        while g_ext_norm > g_tol
            k_old = copy(x_ext[N+1])
            x_old = copy(x_ext[1:N])
            x_ext -= alpha*(hessian_arc(atoms, u_start, xd) \ g_ext)
            x_ext_diff_norm = norm(x_ext0 - x_ext, Inf)

            lala = copy(get_positions(atoms))
            u0_new = x_ext[N+1]*u_start
            u0_old = k_old*u_start
            set_positions!(atoms,lala-u0_old+u0_new)
            x = dofs(atoms)
            set_dofs!(atoms,x-x_old+x_ext[1:N])
            extra_eqn = dot(x_ext-x_ext0,xd) - ds
            g_ext = [gradient(atoms);extra_eqn]
            g_ext_norm = norm(g_ext, Inf)
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

        k = x_ext[N+1]

        res.x_converged = x_ext_diff_norm < x_tol
        res.g_converged = g_ext_norm < g_tol
        res.f_converged = f_diff_norm < f_tol
        res.iteration = iteration
        res.iteration_converged = iteration >= max_iterations

        if verbose > 0
            Minimise.print_res_s(res, g_ext_norm)
        end

        return res , x_ext
    end


end # module