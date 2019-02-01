module BoundaryConditions

    using JuLIP: JVecF, JVecsF, mat, Atoms, set_positions!, get_positions, dofs
    using LsqFit: curve_fit
    using StaticArrays: SVector

    export u_cle, hessian_correction, fit_crack_tip_displacements, intersection_line_plane_vector_scale, location_point_plane_types,
                intersection_line_plane_types, filter_crack_bonds, find_next_bonds_along, find_k, filter_pairs_indices

    """
    `u_cle(atoms::Atoms, tip, K, E, nu) `

    Displacement field, Continuum Linear Elastic solution. Returns displacements in cartesian coordinates.

    Isotropic crack solution:
    ``u_x = C √r [ (2*kappa - 1)cos(θ/2) - cos(3θ/2) ]``
    ``u_y = C √r [ (2*kappa + 1)sin(θ/2) - sin(3θ/2) ]``
    ``C = K / (2√(2pi)E))*(1+nu)``
    ``kappa = 3 - 4nu``

    where the formula for `kappa` is the one for plane strain (plane stress is different).
    
    ### Arguments
    - `atoms::Atoms` or `pos::Array{JVecF}`
    - tip : JVecF
    - K : stress intensity factor
    - E : Youngs modulus
    - nu : Poisson ratio
    """
    function u_cle(pos::Array{JVecF}, tip::JVecF, K, E, nu)

        pos = mat(pos .- [tip])
        x = pos[1,:]
        y = pos[2,:]

        kappa = 3 - 4 * nu
        r = sqrt.(x.^2 + y.^2)
        θ = angle.(x + im * y)
        C = (K / (2*√(2*pi)*E))*(1+nu)

        ux = C*sqrt.(r) .* ((2*kappa-1) * cos.(θ/2) - cos.(3*θ/2))
        uy = C*sqrt.(r) .* ((2*kappa+1) * sin.(θ/2) - sin.(3*θ/2))
        uz = zeros(length(ux))

        return vecs([ux'; uy'; uz'])
    end
    u_cle(atoms::Atoms, tip::JVecF, K, E, nu) = u_cle(get_positions(atoms), tip, K, E, nu)

    """
    `hessian_correction(u::JVecsF, H::SparseMatrixCSC{Float64,Int64}, g::Array{Float64}, idof::Array{Int})`

    Return the corrected displacements, `u_c`, from solving `H \\ g`.

    ### Arguments:
    - `u::JVecsF` : initial displacements
    - `H::SparseMatrixCSC{Float64,Int64}` : hessian, ie `H = hessian(atoms)`
    - `g::Array{Float64}` : gradient, ie `g = gradient(atoms)`
    - `idof::Array{Int}` : degrees of freedom, ie what indices the hessian and gradient where calculated on
    """
    function hessian_correction(u::JVecsF, H::SparseMatrixCSC{Float64,Int64}, g::Array{Float64}, idof::Array{Int})

        u_c = mat(u)[:]
        u_c[idof] -=  H \ g
        u_c = vecs(u_c)

        return u_c
    end

    """
    `hessian_correction(atoms::Atoms, u::JVecsF, idof::Array{Int}; H = nothing, steps = 1)`

    Return the corrected displacements, `u_c` on an atoms objects, from solving `H \\ g`,
    where `H = hessian(atoms)` and `g = gradient(atoms)`.
    Function is able to perform multiple hessian correction steps

    ### Arguments:
    - `atoms::Atoms` : atoms object
    - `u::JVecsF` : initial displacements
    - `idof::Array{Int}` : degrees of freedom, ie what indices the hessian and gradient should be calculated on
    ### Optional Arguments:
    - `H = nothing`: provide hessian (fixed for all steps), defaults to calculating hessian of atoms object for each step
    - `steps = 1`: number of hessian correction steps to compute
    """
    function hessian_correction(atoms::Atoms, u::JVecsF, idof::Array{Int}; H = nothing, steps = 1)

        u_c = JVecsF([])
        u_step = u # for step == 1

        for step in 1:steps
            pos_original = get_positions(atoms)
            set_positions!(atoms, pos_original + u_step)

            if H == nothing H = hessian(atoms) end
            g = gradient(atoms)
            u_step = hessian_correction(u_step, H, g, idof)

            set_positions!(atoms, pos_original) # revert to original positions
            u_c = u_step
        end

        return u_c
    end

    """
    `fit_crack_tip_displacements(atoms::Atoms, atoms_dict, tip_g::JVecF; mask = [1,1,1], verbose = 0)`

    Fit a crack tip using displacements from `Crackcode.BoundaryConditions.u_cle` using a least square method.
    Return the fitted crack tip.

    ### Arguments
    - `atoms::Atoms`
    - `atoms_dict` : should contain :pos_cryst, :K, :E and :nu
    - `tip_g::JVecF` : inital guess for tip position
    - `mask = [1,1,1]` : mask which dimensions, [x,y,z], to fit in/vary  
    - `verbose = 0` : `verbose = 1` returns fitted tip and (`LsqFit.curve_fit`) fit object 
    """
    function fit_crack_tip_displacements(atoms::Atoms, atoms_dict, tip_g::JVecF; 
                                                                        mask = [1,1,1], verbose = 0)
        
        # function which gives new displacements based on a new tip
        function model(x, p) 
                
            # map p into an all 3 dimensions array, then add to initial guess of tip
            pm = Float64.(mask)
            pm[find(mask .== 1)] = p # where the p values should exist if a full dimension array
            tip_p = tip_g + JVecF(pm)

            # calculate new displacements using new tip
            set_positions!(atoms, atoms_dict[:pos_cryst]) 
            u_g = u_cle(atoms, tip_p, atoms_dict[:K], atoms_dict[:E], atoms_dict[:nu]) 
            
            # convert to vector format
            dofs_cryst = dofs(atoms)
            set_positions!(atoms, atoms_dict[:pos_cryst] + u_g)
            dofs_u_g = dofs(atoms) - dofs_cryst

            return dofs_u_g 
        end
        
        # save original positions
        pos_orig = get_positions(atoms)
        
        # obtain vectors of final positions to fit against
        dofs_a = dofs(atoms)
        set_positions!(atoms, atoms_dict[:pos_cryst])
        dofs_cryst = dofs(atoms)
        dofs_u = dofs_a - dofs_cryst;

        # initialise p0 depending on mask
        dim = length(find(mask .== 1))
        p0 = zeros(dim)
        
        # LsqFit.curve_fit
        # model = function which gives new displacements based on a new tip
        # xdata = zeros(similar(dofs_u)), doesn't really matter, just need a vector of the same length
        # ydata = dofs_u, final positions to match
        fit = curve_fit(model, zeros(similar(dofs_u)), dofs_u, p0)
        @printf("fit crack tip using displacements - converged: %s \n", fit.converged)
        
        # map p into an all 3 dimensions array, then add to tip for fitted tip 
        p = fit.param
        pm = Float64.(mask)
        pm[find(mask .== 1)] = p # where the p values should exist in a full dimension array
        tip_f = tip_g + JVecF(pm)
        
        u_orig = pos_orig - atoms_dict[:pos_cryst]
        u_fit = u_cle(atoms, tip_f , atoms_dict[:K], atoms_dict[:E], atoms_dict[:nu])
        @printf("max(norm.('given' u positions - u fit)): %.1e \n", maximum(norm.(u_orig - u_fit)))
        
        # restore original atom positions
        set_positions!(atoms, pos_orig)

        if verbose == 1 return tip_f, fit end
        return tip_f
    end
    
    """
    `intersection_line_plane_vector_scale(p1::SVector{3, Float64}, p2::SVector{3, Float64}, 
    n::SVector{3, Float64}, d::Float64)`
    
    `intersection_line_plane_vector_scale(p1::SVector{3, Float64}, p2::SVector{3, Float64}, 
    n::SVector{3, Float64}, p0::SVector{3, Float64})`

    Check whether line, `p1 + u(p2 - p1)` intersects the plane, `n_x*x + n_y*y + n_z*z + d = 0`.
    Return scale value, u.
    If 0 < u < 1 line intersects plane between x1 and x2. 
    If u = 0, u = 1, line intersects plane on the point x1, x2 respectively.
    If u is Inf, line does not intersect.
    If u is NaN, line is on plane.
    Else line intersects beyond points x1 and x2

    ### Arguments
    - `p1::SVector{3, Float64}, p2::SVector{3, Float64}` : two points on the line
    - `n::SVector{3, Float64}` : normal to plane eg `n_x*x + n_y*y + n_z*z + d = 0`
    - `d::Float64` : plane constant eg `-dot(n, p0)` OR `p0::SVector{3, Float64}` where p0 is a point on the plane
    """
    function intersection_line_plane_vector_scale(p1::SVector{3, Float64}, p2::SVector{3, Float64}, 
                                                            n::SVector{3, Float64}, d::Float64)
        u_n = dot(n, p1) + d
        u_d = dot(n, (p1 - p2))
        u = u_n / u_d

        return u
    end
    function intersection_line_plane_vector_scale(p1::SVector{3, Float64}, p2::SVector{3, Float64}, 
        n::SVector{3, Float64}, p0::SVector{3, Float64})
        return intersection_line_plane_vector_scale(p1, p2, n, -dot(n, p0))
    end

    """
    `location_point_plane_types(atoms::Atoms, indices::Array{Int}, n::JVecF, point_on_plane::JVecF)`

    Sorts/describes the atoms positions into several categories in relation to the given plane.
    Returns boolean lists in regards to the original given indices list.

    ### Example Usage
    to get list of points on the plane => `indices[point_types[:side_on_plane]]`

    ### Arguments
    - `pos::Array{JVecF} or atoms::Atoms`
    - `indices::Array{Int}`
    - `n::JVecF` : normal to plane eg `n_x*x + n_y*y + n_z*z + d = 0`
    - `point_on_plane::JVecF` : point on the plane, to calculate plane constant `-dot(n, p)`

    ### Returns
    - `point_types::Dict` 
        - `:side_with_normal` : on the side in which the normal is pointing
        - `:side_on_plane` : on the plane
        - `:side_inverse_normal` :  on the side in which the sign inverse of the normal is pointing
    """
    function location_point_plane_types(pos::Array{JVecF}, indices::Array{Int}, n::JVecF, point_on_plane::JVecF)

        # calculate equation of plane constant
        d = -dot(n, point_on_plane) 
        
        # initialise empty boolean arrays
        empty_b = Array{Bool}(length(indices))
        [empty_b[i] = false for i in 1:length(empty_b)]
        side_wn = copy(empty_b); side_on = copy(empty_b); side_in = copy(empty_b)

        # compare value from equation of plane and sort
        for i in 1:length(indices)
            ep = dot(n, pos[i]) + d
            if ep > 0.0 side_wn[i] = true
            elseif ep == 0.0 side_on[i] = true
            elseif ep < 0 side_in[i] = true
            end
        end

        point_types = Dict(
            :side_with_normal => side_wn,
            :side_on_plane => side_on,
            :side_inverse_normal => side_in, 
        )

        return point_types
    end
    function location_point_plane_types(atoms::Atoms, indices::Array{Int}, n::JVecF, point_on_plane::JVecF)
        return location_point_plane_types(get_positions(atoms), indices, n, point_on_plane)
    end

    """
    `intersection_line_plane_types(atoms::Atoms, pairs::Array{Tuple{Int, Int}}, n::JVecF, point_on_plane::JVecF)`

    Sorts/describes the line segments, `p_i + u(p_j - pi)`, between given atom pairs into several categories 
    in relation to the given plane `n_x*x + n_y*y + n_z*z + d = 0`.
    Returns boolean lists in regards to the original given pairs list.

    ### Example Usage
    to get list of pairs on the plane => `pairs[pair_types[:side_on_plane]]`

    ### Arguments
    - `pos::Array{JVecF} or atoms::Atoms`
    - `pairs::Array{Tuple{Int, Int}}`
    - `n::JVecF` : normal to plane eg `n_x*x + n_y*y + n_z*z + d = 0`
    - `point_on_plane::JVecF` : point on the plane, to calculate plane constant `-dot(n, p)`

    ### Returns
    - `pair_types::Dict` 
        - `:side_with_normal` : on the side in which the normal is pointing
        - `:side_on_plane` : on the plane
        - `:side_inverse_normal` :  on the side in which the sign inverse of the normal is pointing
        - `:side_crosses_plane` : line segments that cross the plane
        - `:side_on_plane_with_normal` : p_i or p_j is on the plane and the other is on the side with the normal
        - `:side_on_plane_inverse_normal` : p_i or p_j is on the plane and the other is on the inverse side of the normal
    """
    function intersection_line_plane_types(pos::Array{JVecF}, pairs::Array{Tuple{Int, Int}}, n::JVecF, point_on_plane::JVecF)

        # calculate equation of plane constant
        d = -dot(n, point_on_plane)
        
        # initialise empty boolean arrays
        empty_b = Array{Bool}(length(pairs))
        [empty_b[i] = false for i in 1:length(empty_b)]
        side_wn = copy(empty_b); side_on = copy(empty_b); side_in = copy(empty_b)
        side_x = copy(empty_b); side_on_wn = copy(empty_b); side_on_in = copy(empty_b)

        for m in 1:length(pairs)
            p_i = pos[pairs[m][1]]; p_j = pos[pairs[m][2]]
            # get values from equation of plane and sort
            ep_i = dot(n, p_i) + d
            ep_j = dot(n, p_j) + d
            # get intersection value
            u = intersection_line_plane_vector_scale(p_i, p_j, n, d)

            # sort into categories
            if 0.0 < u < 1.0 
                side_x[m] = true 
            elseif u == 0.0 || u == 1.0 
                # need to check positions to seperate which side they are on
                if ep_i == 0
                    if ep_j < 0 side_on_in[m] = true
                    elseif ep_j > 0 side_on_wn[m] = true end
                elseif ep_j == 0
                    if ep_i < 0 side_on_in[m] = true
                    elseif ep_i > 0 side_on_wn[m] = true end
                end
            elseif isinf(u) == true  # line does not intersect plane
                if ep_i > 0 && ep_j > 0 side_wn[m] = true
                elseif ep_i < 0 && ep_j < 0 side_in[m] = true end
            elseif isnan(u) == true side_on[m] = true
            else # line intersects (outside of segment) 
                if ep_i > 0 && ep_j > 0 side_wn[m] = true
                elseif ep_i < 0 && ep_j < 0 side_in[m] = true end
            end
        end

        pair_types = Dict(
            :side_with_normal => side_wn,
            :side_on_plane => side_on,
            :side_inverse_normal => side_in, 
            :side_crosses_plane => side_x,
            :side_on_plane_with_normal => side_on_wn, 
            :side_on_plane_inverse_normal => side_on_in
        )

        return pair_types
    end
    function intersection_line_plane_types(atoms::Atoms, pairs::Array{Tuple{Int, Int}}, n::JVecF, point_on_plane::JVecF)
        return intersection_line_plane_types(get_positions(atoms), pairs, n, point_on_plane)
    end

    """
    `filter_crack_bonds(atoms::Atoms, pair_list::Array{Tuple{Int, Int}}, n_crack_plane::JVecF, point_cp::JVecF,  
    n_crack_front::JVecF, point_cf::JVecF)`

    Return copy pair list with the crack bonds removed. Bonds/line segments between pairs are considered 
    if they all cross the crack plane and
        - are totally behind the crack front
        - cross the crack front
        - are partly on the crack front and behind the crack front
        - on the crack front itself

    ### Arguments
    - `atoms::Atoms`
    - `pair_list::Array{Tuple{Int, Int}}` 
    - `n_crack_plane::JVecF` : normal to plane eg `n_x*x + n_y*y + n_z*z + d = 0`
    - `point_cp::JVecF` : point on the crack plane, `-dot(n, point_cp)`
    - `n_crack_front::JVecF` normal to plane eg `n_x*x + n_y*y + n_z*z + d = 0`
    - `point_cp::JVecF` : point on the crack front, `-dot(n, point_cf)`

    ### Returns
    - `pair_list::Array{Tuple{Int, Int}}` : new pair list with crack bonds removed
    - `mP::Array{JVecF} : mid points of the positions of the pairs
    - `across_crack::Array{Tuple{Int, Int}}` : list of pairs that were removed
    """
    function filter_crack_bonds(atoms::Atoms, pair_list::Array{Tuple{Int, Int}}, 
                                                n_crack_plane::JVecF, point_cp::JVecF,  
                                                n_crack_front::JVecF, point_cf::JVecF)

        # get sorted intersection types of the two planes
        pair_types_cp = intersection_line_plane_types(atoms, pair_list, n_crack_plane, point_cp)
        pair_types_cf = intersection_line_plane_types(atoms, pair_list, n_crack_front, point_cf)

        # combine sets of types
        # get all the pairs that are cross the crack plane
        # and are behind and on the crack front
        cs1 = pair_types_cp[:side_crosses_plane] .* pair_types_cf[:side_inverse_normal]
        cs2 = pair_types_cp[:side_crosses_plane] .* pair_types_cf[:side_crosses_plane] # including ones that across over
        cs3 = pair_types_cp[:side_crosses_plane] .* pair_types_cf[:side_on_plane_inverse_normal]
        cs4 = pair_types_cp[:side_crosses_plane] .* pair_types_cf[:side_on_plane] 

        # generate list of pairs
        across_crack = Array{Tuple{Int, Int}}(0)
        append!(across_crack, pair_list[cs1])
        append!(across_crack, pair_list[cs2])
        append!(across_crack, pair_list[cs3])
        append!(across_crack, pair_list[cs4])
        
        # remove across crack pairs
        pair_list = filter(array -> array ∉ across_crack, pair_list)
        
        # get mid points of new pair list
        P = get_positions(atoms)
        mP = [ 0.5*(P[p[1]] + P[p[2]])  for p in pair_list]
        
        return pair_list, mP, across_crack
    end

    """
    `find_next_bonds_along(atoms::Atoms, pair_list::Array{Tuple{Int, Int}},
                            n_crack_plane::JVecF, point_cp::JVecF, n_crack_front::JVecF, point_cf::JVecF;
                                                    max_pair_separation::Float64 = 0.0, verbose::Int = 0)`


    Return copy pair list with the crack bonds removed. Bonds/line segments between pairs are considered 
    if they all cross the crack plane and
        - are totally behind the crack front
        - cross the crack front
        - are partly on the crack front and behind the crack front
        - on the crack front itself

    ### Arguments
    - `atoms::Atoms`
    - `pair_list::Array{Tuple{Int, Int}}` 
    - `n_crack_plane::JVecF` : normal to plane eg `n_x*x + n_y*y + n_z*z + d = 0`
    - `point_cp::JVecF` : point on the crack plane, `-dot(n, point_cp)`
    - `n_crack_front::JVecF` normal to plane eg `n_x*x + n_y*y + n_z*z + d = 0`
    - `point_cp::JVecF` : point on the crack front, `-dot(n, point_cf)`
    - `max_pair_separation::Float64 = 0.0` : set a max distance of line segments to consider. If 0.0 => will check each pair
    - `verbose::Int = 0`

    ### Returns
    - `next_bonds::Array{Tuple{Int, Int}}` : pair list that are considered the next bond(s)

    if `verbose == 1`, also returns
    - `list_p::Array{Tuple{Int, Int}}` : pair list of all the bonds ahead of the crack front and cross the crack plane

    """
    function find_next_bonds_along(atoms::Atoms, pair_list::Array{Tuple{Int, Int}},
                                    n_crack_plane::JVecF, point_cp::JVecF, n_crack_front::JVecF, point_cf::JVecF;
                                                            max_pair_separation::Float64 = 0.0, verbose::Int = 0)
        
        # get sorted intersection types of the two planes
        pair_types_cp = intersection_line_plane_types(atoms, pair_list, n_crack_plane, point_cp)
        pair_types_cf = intersection_line_plane_types(atoms, pair_list, n_crack_front, point_cf)

        # combine sets of types
        # get all the pairs that arecross the crack plane
        # and are infront of the crack front
        cs1 = pair_types_cp[:side_crosses_plane] .* pair_types_cf[:side_with_normal]
        cs2 = pair_types_cp[:side_crosses_plane] .* pair_types_cf[:side_on_plane_with_normal] # including ones that across over 

        # generate list of pairs
        list_p = Array{Tuple{Int, Int}}(0)
        append!(list_p, pair_list[cs1])
        append!(list_p, pair_list[cs2])
        
        # find pair(s) closest to the crack front
        i_store = Array{Int}(0) # keep indices
        k_tmp = 0.0
        d_cf = -dot(n_crack_front, point_cf)

        for i in 1:length(list_p)
            p1 = atoms[list_p[i][1]]
            p2 = atoms[list_p[i][2]]
            
            # skip pairs with separations larger than given value
            if max_pair_separation != 0.0
                if norm(p2 - p1) > max_pair_separation continue end
            end
            # calculate average value from equation of a plane
            k1 = dot(n_crack_front, p1) + d_cf
            k2 = dot(n_crack_front, p2) + d_cf
            k_av = (k1 + k2)*0.5
            if i == 1 # need to store the very first one
                k_tmp = k_av
                append!(i_store, i)
            end
            if k_av < k_tmp # new lower cost value found
                k_tmp = k_av 
                i_store = Array{Int}(0) # re-initialise / clear store array
                append!(i_store, i)
            elseif k_av == k_tmp # append, pairs with the same value
                append!(i_store, i)
            end
            
        end
        next_bonds = list_p[i_store]
        if length(i_store) == 1 info("next_bonds is an array of a single bond") end
        
        if verbose == 1 return next_bonds, list_p end
        return next_bonds
    end

    # functions required for find_k()
    # script like function, either split or move
    using JuLIP: Atoms, JVecF, JVecsF, minimise!, Exp, get_positions, set_positions!, cutoff, mat
    using ..ManAtoms: separation
    #using CrackCode.BoundaryConditions: u_cle, fit_crack_tip_displacements
    using SciScriptTools.Optimise: bisection_search

    # required for debug parts
    using Logging
    using ..Plot: plot_atoms, plot_bonds, box_around_point
    using ..Potentials: potential_energy
    using ..ManAtoms: atoms_subsystem
    using ASE: ASEAtoms
    using PyPlot: figure, plot, title, axis, legend, vlines, hlines, scatter, xlabel, ylabel, savefig
    """
    `find_k(atoms::Atoms, atoms_dict::Dict, initial_K::Float64, tip::JVecF, 
                        bond_length::Float64, separation_length::Float64, 
                        next_pairs::Array{Tuple{Int, Int}}, across_crack::Array{Tuple{Int, Int}};
                        tip_tol::Float64 = 0.01, bond_length_tol::Float64 = 0.2, separation_tol::Float64 = 0.05, 
                        maxsteps::Int = 10, output_dir::String = "nothing")`

    Self consistently find a good starting stress intensity factor, K, mainly based on fitting the crack tip using
    `CrackCode.BoundaryConditions.fit_crack_tip_displacements()`

    ### Arguments
    - `atoms::Atoms`
    - `atoms_dict::Dict`
    - `initial_K::Float64` : starting guess K # or K_to_u0.ipynb"
    - `tip::JVecF` : crack tip position, where it should exist
    - `bond_length::Float64` : (normal bulk) bond length
    - `separation_length::Float64` : length at which atom pair are considered open/unconnected ie cutoff(calc)
    - `next_pairs::Array{Tuple{Int, Int}}` : pairs that should be 'closed'
    - `across_crack::Array{Tuple{Int, Int}}` : pairs that should be 'open'

    ### Optional Arguments
    - `tip_tol::Float64 = 0.01` : Angstrom value of how close the fitted tip should be
    - `bond_length_tol::Float64 = 0.2` : (scaled percentage) of a bond_length
                            ie `bond_length + bond_length*bond_length_tol`, is considered a optimal bond length
    - `separation_tol::Float64 = 0.05` : (scaled percentage) of a separation_length
                            ie `separation_length - separation_length*separation_tol`, is considered an open bond
    - `maxsteps::Int = 10` : number of iteration attempts
    - `output_dir::String` : location to output debug plots

    ### Returns
    - `K::Float64` : chosen K
    - `u_i::JVecsF` : displacement field at chosen K
    - `u_min::JVecsF` : minimised displacement field at chosen K

    """
    function find_k(atoms::Atoms, atoms_dict::Dict, initial_K::Float64, tip::JVecF, 
                            bond_length::Float64, separation_length::Float64, 
                            next_pairs::Array{Tuple{Int, Int}}, across_crack::Array{Tuple{Int, Int}};
                            tip_tol::Float64 = 0.01, bond_length_tol::Float64 = 0.2, separation_tol::Float64 = 0.05, 
                            maxsteps::Int = 10, output_dir::String = "nothing")

        pos_cryst = get_positions(atoms)
        points = Array{Float64}([initial_K])
        K = points[length(points)]
        u_i = nothing
        u_min = nothing
        directions = nothing
        dir_next = nothing 

        # variables for debug parts
        seps_open_sums = Array{Float64}(0)
        seps_closed_sums = Array{Float64}(0)
        tip_fs = Array{JVecF}(0)
        tip_diffs = Array{Float64}(0)
        if output_dir == "nothing" output_dir = pwd() end # initialise output_dir

        for i in 1:maxsteps

            passes = 0
            dir_next = 0
            K = points[length(points)]
            atoms_dict[:K] = K  # for use in fit_crack_tip_displacements()

            @printf "--- trying K: %.7f \n" K
            set_positions!(atoms, pos_cryst)
            u_i = u_cle(atoms, tip, K, atoms_dict[:E], atoms_dict[:nu])
            set_positions!(atoms, pos_cryst + u_i)
            minimise!(atoms, precond=Exp(atoms, r0=bond_length)) # improvement: should try pass this in
            u_min = get_positions(atoms) - pos_cryst

            # main condition for determining search for K
            # fit crack tip (in x and y) to compare later
            tip_f = nothing
            tip_f = fit_crack_tip_displacements(atoms, atoms_dict, tip, mask=[1,1,0])
            tip_diff_x = abs(tip[1] - tip_f[1])
            @printf "Difference in given tip and fitted tip in x %.7f\n" tip_diff_x
            if tip_diff_x <= tip_tol
                @printf "fitted tip is within tolerance\n"
                passes += 1
            end
            push!(tip_fs, tip_f)
            push!(tip_diffs, tip_diff_x)

            # across crack check
            debug("How many pairs across the crack are open?")
            seps_u_min = separation( pos_cryst + u_min, across_crack )
            seps_open = find(seps_u_min .> separation_length - separation_tol)
            debug("  Open Pairs: ", length(seps_open))
            debug("Num of Pairs: ", length(across_crack))
            debug("Min separation of across_crack: ", minimum(seps_u_min))
            # they should all be open
            if length(seps_open) == length(across_crack) 
                @printf "across crack pairs are all open\n"
                passes += 1
            end

            seps_open_sum = sum(seps_u_min)
            push!(seps_open_sums, seps_open_sum)

            # next bond check
            debug("How many next pairs are closed?")
            seps_u_min_next_pairs = separation( pos_cryst + u_min, next_pairs )
            seps_next_pairs_closed = find(seps_u_min_next_pairs .< bond_length + bond_length_tol)
            debug("Closed Pairs: ", length(seps_next_pairs_closed))
            debug("Num of Pairs: ", length(next_pairs))
            debug("Max separation of next_pairs: ", maximum(seps_u_min_next_pairs))

            # they should all be closed
            if length(seps_next_pairs_closed) == length(next_pairs)
                @printf "next pairs are all closed\n"
                passes += 1
            end

            seps_closed_sum = sum(seps_u_min_next_pairs)
            push!(seps_closed_sums, seps_closed_sum)

            # plot system and tip
            if Logging.configure().level == DEBUG
                figure()
                plot_atoms(atoms, colour = "grey")
                plot_bonds(atoms, across_crack, label = "pairs: across crack")
                plot_bonds(atoms, next_pairs, colour="red", label = "pairs: next")
                t_x = tip[1] ; t_y = tip[2]
                plot(t_x, t_y, "o", markersize = 4, label = "tip : target")
                tf_x = tip_f[1] ; tf_y = tip_f[2]
                plot(tf_x, tf_y, "o", markersize = 4, label = "tip : fitted")
                title("K : $(round(K, 7))")
                axis(box_around_point([t_x, t_y], [10,10]))
                legend()
                savefig(joinpath(output_dir, "debug_system_tip.pdf"))
                close()
            end

            # ideal situation
            # tip within tip tolerance
            # across_crack pairs all open within tolerance
            # next_ pairs all closed within tolerance
            if passes >= 3 break end

            # tip bisection search has convergence to be within the tip tolerance for past steps
            if length(find(tip_diffs .< tip_tol)) >= 5
                @printf "tip convergenced to be within the tiptolerance for past 5 iterations \n"
                break
            end

            @printf "--- What to do with K?\n"
            if (tip[1] - tip_f[1]) > 0.0 dir_next = 1 # if crack is closing up
            elseif (tip[1] - tip_f[1]) < 0.0 dir_next = -1 end #if crack is opening up

            if dir_next == 1 @printf "increasing K for next loop\n"
            elseif dir_next == -1 @printf "decreasing K for next loop\n" end

            # search scale increment (when not bisecting)
            bond_safe = bond_length + bond_length_tol
            diff_per = (maximum(seps_u_min_next_pairs) - bond_safe) / bond_safe
            search_scale = abs(diff_per)

            K, points, directions = bisection_search(points, dir_next, directions, search_scale = search_scale )

        end

        # plot simple potenital (using dimer) with length tolerances
        if Logging.configure().level == DEBUG
            figure()
            r = collect(linspace(bond_length*0.8, cutoff(atoms.calc)*1.2, 100))
            atoms_pe = Atoms(atoms_subsystem(ASEAtoms(atoms), [1,2]))
            pe = potential_energy(atoms_pe, atoms.calc, r)
            plot(r, pe, color = "b", label = "simple dimer potential")
            xlabel("Dimer Separation Distance, r")
            ylabel("Energy, eV")
            vlines(bond_length*(1-bond_length_tol), minimum(pe), maximum(pe), 
                                            color = "orange", linestyle="dashed", label = "bond_length tolerance")
            vlines(bond_length*(1+bond_length_tol), minimum(pe), maximum(pe), 
                                            color = "orange", linestyle="dashed")
            vlines(separation_length*(1-separation_tol), minimum(pe), maximum(pe), linestyle="dashed", 
                                            color = "red", label = "separation_length tolerance")
            vlines(separation_length*(1+separation_tol), minimum(pe), maximum(pe), linestyle="dashed", 
                                            color = "red",)
            title("Simple Dimer Overlayed with Length Tolerances")
            legend()
            savefig(joinpath(output_dir, "debug_dimer_overlayed_with_length_tolerances.pdf"))
            close()
        end   

        # plot difference for given tip and fitted tip
        if Logging.configure().level == DEBUG
            figure()
            iters = collect(1:length(tip_diffs))
            plot(iters, tip_diffs, "o-")
            hlines(tip_tol, minimum(iters), maximum(iters), linestyle="dashed", color = "grey", label = "tip_tol $tip_tol")
            xlabel("Iteration")
            ylabel("abs( tip_given - tip_fitted )")
            title("Tip Movement Convergence")
            legend()
            savefig(joinpath(output_dir, "debug_tip_movement_convergence.pdf"))
            close()
        end

        # plot sum of separations of next_pairs and across crack
        if Logging.configure().level == DEBUG
            figure()
            x = mat(tip_fs)[1,:]
            y = seps_closed_sums
            scatter(x, y, label = "different systems wrt K")
            index = length(points)-1
            scatter(x[index], y[index], color = "red", label = "chosen system wrt K")
            optimal_sum = bond_length*length(next_pairs)
            hlines(optimal_sum, minimum(x), maximum(x), linestyles="dashed", color="orange", label = "optimal sum")
            vlines(tip[1], minimum(y), maximum(y), linestyles="dashed", color="grey", label = "target tip")
            xlabel("Crack Tip Position in x")
            ylabel("Sum of separations of next_pairs")
            title("Optimal Sum of Separation of next_pairs wrt K and tip")
            legend()
            savefig(joinpath(output_dir, "debug_optimal_separation_of_next_pairs.pdf"))
            close()

            figure()
            x = mat(tip_fs)[1,:]
            y = seps_open_sums
            scatter(x, y, label = "different systems wrt K")
            index = length(points)-1
            scatter(x[index], y[index], color = "red", label = "chosen system wrt K")
            minimum_sum = cutoff(atoms.calc)*length(across_crack)
            hlines(minimum_sum, minimum(x), maximum(x), linestyles="dashed", color="orange", label = "minimum sum")
            vlines(tip[1], minimum(y), maximum(y), linestyles="dashed", color="grey", label = "target tip")
            xlabel("Crack Tip Position in x")
            ylabel("Sum of separations of across_crack")
            title("Minimum Sum of Separation of across_crack wrt K and tip")
            legend()
            savefig(joinpath(output_dir, "debug_minimum_separation_of_across_crack.pdf"))
            close()
        end  

        if length(points)-1 == maxsteps @printf "maxsteps: %d reached\n" maxsteps end
        return K, u_i, u_min
    end

    """
    `filter_pairs_indices(pair_list::Array{Tuple{Int, Int}}, indices::Array{Int})`

    Filter pairs that are associated with particular indices

    ### Arguments
    - `pair_list::Array{Tuple{Int, Int}}` : list of pairs
    - `indices::Array{Int}` : list of atoms indices to exclude from list of pairs
    """
    function filter_pairs_indices(pair_list::Array{Tuple{Int, Int}}, indices::Array{Int})
        remove_indices = Array{Int}(0)
        for pi in 1:length(pair_list)
            p_i = pair_list[pi][1] ; p_j = pair_list[pi][2]
            if length(find(p_i .== indices)) >= 1 push!(remove_indices, pi)
            elseif length(find(p_j .== indices)) >= 1 push!(remove_indices, pi) end
        end

        keep_indices = filter(array -> array ∉ remove_indices, linearindices(1:length(pair_list)))

        return pair_list[keep_indices]
    end

    function filter_pairs_indices(pair_list::Array{Tuple{Int, Int}}, index::Int)
        return filter_pairs_indices(pair_list, [index])
    end
    # could filter pairs by length


### Old code

#using JuLIP
#using PyCall

using JuLIP: JVecF, Atoms, AbstractAtoms, mat, vecs, set_pbc!, get_pbc, get_positions, set_positions!, get_cell, set_cell!,
                cutoff
using NeighbourLists: PairList
#using ASE.MatSciPy: NeighbourList

using JuLIP.Preconditioners: Exp

# maybe keep if I need angle information
# TODO: rewrite in style of newer functions
function radial_positions(atoms, point)

    tip_x = point[1]
    tip_y = point[2]
    tip_z = point[3]

    x = mat(positions(atoms))[1, :]
    y = mat(positions(atoms))[2, :]
    z = mat(positions(atoms))[3, :]
    xp, yp, zp = x - tip_x, y - tip_y, z - tip_z
    r = sqrt.(xp.^2 + yp.^2 + zp.^2)
    theta = atan2.(yp, xp)
    phi = acos.(zp ./ r)

    return r, theta, phi
end

function midpoints(atoms, pair_list)

    pos = get_positions(atoms)
    mB = [ 0.5*(pos[pair[1]] + pos[pair[2]])  for pair in pair_list]

    return mB
end

# why is the +1e-15 needed? also works for 0.01 and inbetween, not 1e-16
# cutoff(calc) is already 1.2 and a0 is 1.0
"""
    get_bonds(atoms::AbstractAtoms; periodic_bc = nothing )

Compute list of bonds of atoms object.
`periodic_bc` allows for modification of pbc just for calculating bonds.
Returns `bonds_list` and `mB` midpoint of each bond.
"""
function get_bonds(atoms::AbstractAtoms; periodic_bc = nothing )

    # not tested in actual 3D
    # may need to change how to actually deal with combinations of true/false pbc, (during sort?)

    original_pbc = get_pbc(atoms)

    if periodic_bc != nothing
        set_pbc!(atoms, periodic_bc)
    end

    pos = get_positions(atoms)
    bonds_list = Tuple{Int,Int}[]

    # something doesnt get recalculated when calling get_bonds on an atoms object that it has been called on
    # might be bonds( ... ) below
    nlist = PairList(pos, cutoff(atoms.calc), get_cell(atoms), get_pbc(atoms))
    for k in 1:length(nlist.i)
        i = nlist.i[k] 
        j = nlist.j[k]
        if i < j
            push!(bonds_list, (i,j))
        end
    end

    pos = get_positions(atoms)
    mB = [ 0.5*(pos[pair[1]] + pos[pair[2]])  for pair in bonds_list]

    set_pbc!(atoms, original_pbc)

    # atoms.transient is set when calling bonds(atoms, cutoff(atoms.calc)+1e-15)
    # why delete? when calling again it won't recalculate and thus use the same bonds list
    #delete!(atoms.transient, (:nlist, cutoff(atoms.calc)+1e-15))

    return bonds_list, mB
end

"""
    get_bonds(atoms::AbstractAtoms, index::Int; bonds_list = nothing)

Copmute/find bonds connected to a sepcifc atoms
Return bonds_list type object
"""
function get_bonds(atoms::AbstractAtoms, index::Int; bonds_list = nothing)

    if bonds_list == nothing
        bonds_list, _ = get_bonds(atoms)
    end

    # for cases when index appears as i or j in bonds_list
    indices_list = find(index .== [b[1] for b in bonds_list])
    indices_list_j = find(index .== [b[2] for b in bonds_list])

    append!(indices_list, indices_list_j)

    mB_indices_list = midpoints(atoms, bonds_list[indices_list])

    # not much need for indices_list, this is the indices of the bonds of the new list in the given bonds list
    # but used to filter the original bonds_list, maybe just return one?
    # eg where bonds_list is the original bonds_list, are the same
    # CrackCode.Plot.plot_bonds(atoms, bonds_list, indices=list_a1)
    # CrackCode.Plot.plot_bonds(atoms, bonds_list_a1)
    return bonds_list[indices_list], mB_indices_list, indices_list

end

"""
    build_boundary_clamp(atoms, thickness=false)

Returns list of indices.
Region to fix during minimisations of (deafult) thickness of `cutoff(atoms.calc)`
"""
function build_boundary_clamp(atoms, thickness=false)

    if thickness == false
        thickness = cutoff(atoms.calc)
    end

    pos = mat(get_positions(atoms))
    X = pos[1,:]
    Y = pos[2,:]
    xmin, xmax = extrema(X)
    ymin, ymax = extrema(Y)
    Iclamp = find( (X .<= xmin + thickness) + (X .>= xmax - thickness)
                   + (Y .<= ymin + thickness) + (Y .>= ymax - thickness)  )

    # no longer in JuLIP, commented out due to bug, https://github.com/libAtoms/JuLIP.jl/issues/85
    #set_data!(atoms, :Iclamp, [Iclamp])
    return Iclamp
end

"""
    fix_neighbourlist(atoms, bonds_list)

Uses `set_transient` to fix the neighbour list
To remove delete!(atoms.transient, (:nlist, cutoff(calc)))
"""
function fix_neighbourlist(atoms, bonds_list)

    P = get_positions(atoms)
    i1 = [b[1] for b in bonds_list]
    j1 = [b[2] for b in bonds_list]
    i = [i1; j1]
    j = [j1; i1]
    R = [ JVecF(P[jj] - P[ii]) for (ii, jj) in zip(i, j) ]
    r = norm.(R)
    s = zeros(JVec{Int32}, 2*length(bonds_list))

    # generate neighbour list from D.B and stcutoff(calc),ore in Atoms
    nlist = JuLIP.ASE.MatSciPy.NeighbourList(cutoff(atoms.calc), i, j, r, R, s, nothing, 2*length(bonds_list))
    set_transient!(atoms, (:nlist, cutoff(atoms.calc)), nlist, Inf)

    return nlist
end

"""
`u_relaxed(atoms, u, constraints)`

Minimise atoms and return relaxed displacements, u_r.
# Returns
- `u_r`: relaxed displacements
# Arguments
- `atoms`: AbstractAtoms object
- `u`: intial displacements
- `constraints`: for minimisation
"""
function u_relaxed(atoms, u, constraints)

    atoms_min = deepcopy(atoms)
    pos_original = get_positions(atoms)

    set_constraint!(atoms_min, constraints)
    set_positions!(atoms_min, pos_original + u)

    # FIX need to change Exp and r0!!
    minimise!(atoms_min, gtol=1e-5, precond=Exp(atoms_min, r0=1.0))

    # relaxed displacements
    u_r = get_positions(atoms_min) - pos_original

    return u_r
end

# Feature: function should be editted to handle no minimise ie solution_steps = 0
# for later on, when we can't minimise, but still want to use the multi correction parts
"""
    u_solution(atoms, u_initial, Idof, constraints, free;
                        correction_steps_per_solution = 1, solution_steps = 1)

u_solution is the displacements from crystal positions such that, the free atoms given

Returns seperation of pair of atoms.
# Returns
- `u_sol`: corrected and relaxed displacements
# Arguments
- `atoms`: AbstractAtoms object
- `u_inital`: intial displacements for hessian correction
- `Idof`: degreess of freedom for hessian correction
- `steps`: number of hessian steps to compute before relaxing system
- `constraints`: for minimisation
- `free`: indices free to move
- `correction_steps_per_solution`: number of hessian correction steps to compute
- `solution_steps`: number of times to minimise the system and calculate solution
"""
function u_solution(atoms, u_initial, Idof, constraints, free;
                        correction_steps_per_solution = 1, solution_steps = 1)

    u_sol = []
    u_corr = []
    u_i = []
    for i in 1:solution_steps

        if i == 1
            u_i = u_initial
        end

        # perform correction step if arguement is => 1, otherwise don't
        if correction_steps_per_solution >= 1
            u_corr = hessian_correction(atoms, u_i, Idof, steps = correction_steps_per_solution)
        elseif correction_steps_per_solution == 0
            u_corr = u_initial
        end

        u_sol = copy(u_corr)
        u_r = u_relaxed(atoms, u_corr, constraints)

        # could get rid of arguement, `free`, using line below
        # only tested using FixedCell constraint, might not work when fixing bond length etc
        #free = constraints.ifree[3:3:end] ÷ 3
        u_sol[free] = u_r[free]
        u_i = u_sol
    end

    return u_sol, u_corr
end



end
