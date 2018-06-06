# Manlipulation of atoms object
# or other atoms object properties

module ManAtoms

    using JuLIP: JVecF, Atoms, get_positions, set_positions!, mat, atomdofs, energy, gradient, hessian, dofs, set_dofs!
    using ASE: ASEAtoms
    using ConstrainedOptim: TwiceDifferentiable, TwiceDifferentiableConstraints, IPNewton, optimize
    using Optim: Options
    using ForwardDiff #using ForwardDiff: hessian

    export seperation, dimer, atoms_subsystem, mask_atom!, move_atom_pair!, pair_constrained_minimise!

    """
    `separation(atoms::Atoms, i::Int, j::Int) `

    Returns norm distance between pair of positions.

    ### Arguments
    - `atoms::Atoms` or `pos::Array{JVecF}`
    - i::Int, j::Int or pair::Tuple{Int, Int} or pairs::Array{Tuple{Int, Int}} : atom pair(s)
    """
    separation(pos::Array{JVecF}, i::Int, j::Int) = norm(pos[j] - pos[i])
    separation(pos::Array{JVecF}, pair::Tuple{Int, Int}) = separation(pos, pair[1], pair[2])
    separation(atoms::Atoms, i::Int, j::Int) = separation(get_positions(atoms), i, j)
    separation(atoms::Atoms, pair::Tuple{Int, Int}) = separation(atoms, pair[1], pair[2])

    function separation(pos::Array{JVecF}, pairs::Array{Tuple{Int, Int}})
        len = length(pairs)
        seps = zeros(len)
        for n in 1:len
            seps[n] = separation(pos, pairs[n])
        end
        return seps
    end
    separation(atoms::Atoms, pairs::Array{Tuple{Int, Int}}) = separation(get_positions(atoms), pairs)

    """
    `systemsize(atoms::Atoms)`
    `systemsize(pos::Array{JVecF})`

    Obtain the size of the system using the extreme positions of the atoms in each direction (rather than cell size).
    """
    function systemsize(pos::Array{JVecF})
        pos_m = mat(pos)
        x = pos_m[1,:]
        y = pos_m[2,:]
        z = pos_m[3,:]
        x_max = maximum(x); x_min = minimum(x)
        y_max = maximum(y); y_min = minimum(y)   
        z_max = maximum(z); z_min = minimum(z)
    
        systemlengths = [x_max - x_min, y_max - y_min, z_max - z_min]
        return systemlengths
    end
    systemsize(atoms::Atoms) = systemsize(get_positions(atoms))


    """
    `dimer(element = "H"; seperation = 1.0, cell_size = 30.0)`

    Build a dimer object with specifc seperation and cell size

    ### Arguments
    - element : chemical element
    - seperation : distance between atoms along x axis
    - cell_size : (same in all directions)
    """
    function dimer(element = "H"; seperation = 1.0, cell_size = 30.0)

        a_string = join([element, "2"])
        atoms = Atoms(ASEAtoms(a_string))
        set_cell!(atoms, cell_size*eye(3))
        set_pbc!(atoms, true)
        pos = mat(get_positions(atoms))
    
        pos[1,1] += -seperation/2
        pos[1,2] += +seperation/2
    
        set_positions!(atoms, pos)
    
        return atoms
    end

    """
        `atoms_subsystem(atoms, indices)`

    Returns a new atoms object, of given indices, carved out of the given atoms object.
    Not super efficient memory wise!

    ### Arguments
    - `atoms `: atoms object
    - `indices`: list of atom indices
    """
    function atoms_subsystem(atoms::ASEAtoms, indices)

        # copy and cut out rest of system
        atoms_sub = deepcopy(atoms)
        indices_inverse = setdiff(1:length(atoms), indices)
        indices_inverse_py = indices_inverse .- 1
        delete!(atoms_sub.po, indices_inverse_py)

        return atoms_sub
    end

    """
    `move_atom_pair!(atoms::Atoms, pairs::Array{Tuple{Int, Int}}, separations::Array{Float64})`

    Manually move atom pairs to satisfy given separations. Default tries to move them evenly.
    Note: Order of pair list matters for recurring indices. 
    First occurrence will be moved evenly, second occurrence will not.
    
    ### Agruments
    - `atoms::Atoms``
    - `pairs::Array{Tuple{Int, Int}}` : list of atom pairs
    - `separations::Array{Float64}` : list of separations
    
    ### Other Usage
    `move_atom_pair!(atoms::Atoms, pair::Tuple{Int, Int}, seperation::Float64; i_s = 0.5, j_s = 0.5)`
    - `i_s = 0.5` : amount to scale atom 1 position 
    - `j_s = 0.5` : amount to scale atom 2 position 
    """
    # manually move atom pair to satisfy given seperation
    function move_atom_pair!(pos::Array{JVecF}, i::Int, j::Int, seperation::Float64; i_s = 0.5, j_s = 0.5)
        u = pos[j] - pos[i]
        v = 1.0 - seperation / norm(u)
        pos[i] += i_s * v * u
        pos[j] -= j_s * v * u
        return pos
    end
    move_atom_pair!(pos::Array{JVecF}, pair::Tuple{Int, Int}, seperation::Float64; i_s = 0.5, j_s = 0.5) = 
                                                                move_atom_pair!(pos, pair[1], pair[2], seperation, i_s = i_s, j_s = j_s)   
    function move_atom_pair!(atoms::Atoms, i::Int, j::Int, seperation::Float64; i_s = 0.5, j_s = 0.5)
        pos = get_positions(atoms)
        pos = move_atom_pair!(pos, i, j, seperation, i_s = i_s, j_s = j_s)
        set_positions!(atoms, pos)
        return atoms
    end
    move_atom_pair!(atoms::Atoms, pair::Tuple{Int, Int}, seperation::Float64; i_s = 0.5, j_s = 0.5) = 
                                                            move_atom_pair!(atoms, pair[1], pair[2], seperation, i_s = i_s, j_s = j_s)
    
    # atoms and array of pairs and separations    
    function move_atom_pair!(atoms::Atoms, pairs::Array{Tuple{Int, Int}}, separations::Array{Float64})
        
        i_s = 0.0 
        j_s = 0.0
        
        pos = get_positions(atoms)
        indices_moved = []
        for i in 1:length(pairs)
            m_1 = findin(indices_moved, pairs[i][1])
            m_2 = findin(indices_moved, pairs[i][2])
    
            if length(m_1) == 0 && length(m_2) == 0 i_s = 0.5; j_s = 0.5 end
            if length(m_1) >= 1 i_s = 0.0; j_s = 1.0 end
            if length(m_2) >= 1 i_s = 1.0; j_s = 0.0 end
    
            # they are done so skip
            if length(m_1) >= 1 && length(m_2) >= 1
                warn("Both atoms, ", pairs[i], ", have already been moved - can not satisfy seperation")
                continue
            end
    
            pos = move_atom_pair!(pos, pairs[i], separations[i], i_s = i_s, j_s = j_s)
    
            # add to done list
            push!(indices_moved, pairs[i][1])
            push!(indices_moved, pairs[i][2])
        end
        set_positions!(atoms, pos)
        return atoms
    end

    """
    `pair_constrained_minimise!(atoms::Atoms, pairs::Array{Tuple{Int, Int}}, seperations::Array{Float64}; 
                                            s_tol = 1e-3, g_tol = 1e-2)`

    Note: Separation distances between atom pairs should initially be satisfied. (Warning is produced)

    ### Arguments
    - `atoms::Atoms`
    - `pairs::Array{Tuple{Int, Int}}` : list of atom pairs
    - `seperations::Array{Float64}` : list of separation distances
    - `s_tol = 1e-3` : separation tolerance
    - `g_tol = 1e-2` : gradient tolerance - Optim optimize option https://github.com/JuliaNLSolvers/Optim.jl/blob/master/docs/src/user/config.md
    """
    function pair_constrained_minimise!(atoms::Atoms, pairs::Array{Tuple{Int, Int}}, seperations::Array{Float64}; 
                                            s_tol = 1e-3, g_tol = 1e-2)
        
        # get associated degrees of freedom for the pairs
        dofspairs = []
        dofspairjoined = []
        for pair in pairs
            p = atomdofs(atoms, pair[1])
            q = atomdofs(atoms, pair[2])
            push!(dofspairs, (p, q))
            r = copy(p)
            append!(r, q)
            push!(dofspairjoined, r)
        end

        # separation between atoms
        slen(x, Ip, Iq) = norm(x[Iq] - x[Ip])

        # energy, gradient and hessian for the interatomic potential
        ener(x) = energy(atoms, x)
        gradient!(g, x) = (g[:] = gradient(atoms, x); g)
        hessian!(h, x) = (h[:,:] = hessian(atoms, x); h)

        # constraint function
        function con_c!(c, x) 
            for i in 1:length(dofspairs)
                c[i] = slen(x, dofspairs[i][1], dofspairs[i][2]) - seperations[i]
            end
            return c
        end

        # Jacobian of constraint, shape [1, length(x)]
        function con_jacobian!(j, x)
            j[1,:] = 0.0
            for i in 1:length(dofspairs)   
                pair_len = slen(x, dofspairs[i][1], dofspairs[i][2])
                j[1,dofspairs[i][1]] = (x[dofspairs[i][1]]-x[dofspairs[i][2]])/pair_len
                j[1,dofspairs[i][2]] = (x[dofspairs[i][2]]-x[dofspairs[i][1]])/pair_len
            end
            return j
        end

        function con_h!(h, x, λ)
            for i in 1:length(dofspairjoined)
                p = length(dofspairs[i][1])
                q = length(dofspairs[i][2])
                _i1 = 1:p
                _i2 = p+1:p+q
                _cf(x) = norm(x[_i2] - x[_i1]) - seperations[i]
                h[dofspairjoined[i], dofspairjoined[i]] += 
                                            λ[1] *ForwardDiff.hessian(_cf, x[dofspairjoined[i]])
            end
            return h
        end

        x = dofs(atoms)
        df = TwiceDifferentiable(ener, gradient!, hessian!, x)
        lx = Float64[]; ux = Float64[]
        lc = fill(-s_tol, length(pairs)); uc = fill(s_tol, length(pairs))
        dfc = TwiceDifferentiableConstraints(con_c!, con_jacobian!, con_h!,
                                            lx, ux, lc, uc)

        res = optimize(df, dfc, x, IPNewton(show_linesearch=false, μ0=:auto), 
                        Options(show_trace = true, extended_trace = false,
                                        allow_f_increases = true, successive_f_tol = 2,
                                        g_tol = g_tol))
        set_dofs!(atoms, res.minimizer)

        return res
    end

    """
    `mask_atom!(mask::Array{Bool,2}, index::Int) `
    `mask_atom!(mask::Array{Bool,2}, indices::Array{Int,1})`

    Returns the given mask, with the given indices of atoms masked out, ie equal to 0
    """
    function mask_atom!(mask::Array{Bool,2}, index::Int) 
        mask[:, index] = 0
        return mask
    end
    function mask_atom!(mask::Array{Bool,2}, indices::Array{Int,1})
        for i in 1:length(indices)
            mask[:, indices[i]] = 0
        end
        return mask
    end

    """
        `do_w_mod_pos(some_function::Function, atoms::Atoms, pos)`

    "Do with modified positions"
    Returns the given function's output given an atoms object and its modified positions

    ### Usage
    - `do_w_mod_pos(forces, atoms, pos + u)`

    ### Arguments
    - `do_function::Function`: some function that takes an atoms object
    - `atoms::Atoms`: atoms object
    - `pos`: positions

    TODO:
    - have arguments for the function
    - pass multiple functions to modify atoms and then get propery
        - ie have a overall do_mod_atoms function?
    - macro may help?
    """
    function do_w_mod_pos(do_function::Function, atoms::Atoms, pos)
        # save and set modified positions
        pos_original = get_positions(atoms)
        set_positions!(atoms, pos)

        fun_output = do_function(atoms)

        # revert to initial positions
        set_positions!(atoms, pos_original)

        return fun_output
    end

# Old code left for reference and eventually clean up

using JuLIP
using PyCall
using PyPlot

using ASE

#export centre_point

include("Generic.jl")

# ----- strucutre -----

"""
Extend JuLIP.ASE.ASEAtoms object
to be able to add data of any size/length and associate with atoms object
(rather than set_array/set_data which has to equal number of atoms)
"""
type atoms_extended
    atoms::ASEAtoms
    data_extended::Dict{}
end

# initialisation - basic
function atoms_extended(atoms::ASEAtoms)
    data_extended = Dict()
    return atoms_extended(atoms, data_extended)
end

# getters and setters
# not much point of these anymore
# maybe if I change the strucutre of the type?
# consistency?
function set_data_extended!(at_ex::atoms_extended, key, data)
    at_ex.data_extended[key] = data
end

function get_data_extended(at_ex::atoms_extended, key)
    data = at_ex.data_extended[key]
    return data
end





# ----- build atoms -----

"""
    triangular_lattice_slab(a, N_x, N_y)

Build a traingular lattice slab using matscipy idealbrittlesolid

#### Arguments
- a : bondlength
- N_x : size of system in x direction
- N_y : size of system in y direction
- shift_to_centre : shift positions such that bond mid point is at cell y midpoint
#### Notes
Use 2*N x N for a rough square system
"""
function triangular_lattice_slab(a, N_x, N_y, ; shift_to_centre = false)

    @pyimport matscipy.fracture_mechanics.idealbrittlesolid as ibs

    atoms = ASEAtoms(ibs.triangular_lattice_slab(a, N_x, N_y))

    if shift_to_centre == true
      pos = mat(get_positions(atoms))
      # mid_point in height between pair is 0.5 ( a0 * cos(pi/6) )
      pos[2,:] += a*sqrt(3)/4
      set_positions!(atoms, pos)
    end

    return atoms
end

"""
    centre_point(atoms)

Centre point of cell of atoms objecteturns x, y, z
"""
function cell_centre_point(atoms)

    x = get_cell(atoms)[1]/2
    y = get_cell(atoms)[5]/2
    z = get_cell(atoms)[9]/2

    return [x, y, z]
end

# ----- properties of atoms -----

"""
Calculate tip position of an centred idealbrittlesolid
"""
function calc_tip_idealbrittlesolid(atoms, a0)

  #if object is centred
  tip_x, tip_y, tip_z = diag(get_cell(atoms))/2

  # NOTE: this shift has been moved to when one builds the slab triangular_lattice_slab()
  # mid point in y direction along bonded pair
  # mid_point = 0.5 ( a0 * cos(pi/6) )
  # tip_y -= a0*(sqrt(3)/4)

  #println("tip = (", tip_x, ", ", tip_y, ")")

  return [tip_x, tip_y, tip_z]
end


"""
Calculate radial positions from given point
"""
function calc_radial_positions_xy(atoms, point)

    tip_x = point[1]
    tip_y = point[2]

    x = mat(get_positions(atoms))[1, :]
    y = mat(get_positions(atoms))[2, :]
    xp, yp = x - tip_x, y - tip_y
    r = sqrt(xp.^2 + yp.^2)
    t = atan2(yp, xp)
    return r, t
end

function build_edge_boundary_clamp(atoms, thickness=3.0)
    X = mat(get_positions(atoms))
    # rcut = cutoff(idealbrittlesolid)
    x, y = X[1,:], X[2,:]
    Iclamp = find(
        (x .< minimum(x) + thickness) + (x .> maximum(x) - thickness) +
        (y .< minimum(y) + thickness) + (y .> maximum(y) - thickness) )

    return Iclamp

end

function find_tip_coordination(atoms, bondlength, bulk_nn, groups)
    # groups : boolean array of indices to ignore
    #          0 ignore, 1 include

    nb_list = NeighbourList(atoms, bondlength)
    neighbours_counted = generic.bincount(atoms, nb_list.i)
    #println("neighbours_counted:", neighbours_counted)
    #println("length(neighbours_counted:", length(neighbours_counted))
    indices = linearindices(atoms)

    y = mat(get_positions(atoms))[2,:]

    cell_y_mid = get_cell(atoms)[5]/2
    #println("cell_y_mid:", cell_y_mid)
    unbonded_bool = neighbours_counted .< bulk_nn
    above_bool = y .> cell_y_mid
    below_bool = y .< cell_y_mid
    above_unbonded_bool = unbonded_bool & above_bool
    nongroup_bool = (indices .!= groups)

    above_unbonded_nongroup = (above_bool & unbonded_bool) & nongroup_bool
    below_unbonded_nongroup = (below_bool & unbonded_bool) & nongroup_bool

    above_indices = indices[above_unbonded_nongroup]
    below_indices = indices[below_unbonded_nongroup]

    x = mat(get_positions(atoms))[1,:]

    bond_1 = above_indices[findmax(x[above_indices])[2]]
    bond_2 = below_indices[findmax(x[below_indices])[2]]

    #above_unbonded_indices = indices[(above_bool & unbonded_bool)]
    #below_unbonded_indices = indices[(below_bool & unbonded_bool)]

    return bond_1, bond_2, above_indices, below_indices, cell_y_mid
    #return bond_1, bond_2
end

function get_seperation(atoms, index_1, index_2)

    pos = mat(get_positions(atoms))
    pos_i1 = pos[:,index_1]
    pos_i2 = pos[:,index_2]
    seperation = sqrt((pos_i1[1]-pos_i2[1])^2 + (pos_i1[2]-pos_i2[2])^2 + (pos_i1[3]-pos_i2[3])^2)

    return seperation
end

function get_size(atoms)

    pos = mat(get_positions(atoms))

    x_length = maximum(pos[1,:]) - minimum(pos[1,:])
    y_length = maximum(pos[2,:]) - minimum(pos[2,:])
    z_length = maximum(pos[3,:]) - minimum(pos[3,:])

    return [x_length, y_length, z_length]
end

function calc_radial_size_xy(atoms)

    system_size = get_size(atoms)
    radial_size = system_size/2
    radial_size_xy = minimum([radial_size[1], radial_size[2]])

    return radial_size_xy
end

# ----- manlipulation atoms -----

function align_to_point(atoms, point)

  x = point[1]
  y = point[2]

  #atoms_centre_point = get_size(atoms)/2
  atoms_centre_point = diag(cell(atoms))/2.0

  atoms_aligned = deepcopy(atoms)
  new_positions = mat(get_positions(atoms))
  new_positions[1,:] += x - atoms_centre_point[1]
  new_positions[2,:] += y - atoms_centre_point[2]
  set_positions!(atoms_aligned, new_positions)
  # seems redundent
  set_cell!(atoms_aligned, get_cell(atoms))

  return atoms_aligned

end


"""
Generates new atoms of atoms_2 aligned to centre of atoms_1
"""
function align_centres(atoms_1, atoms_2)

    centre_point_1 = cell_centre_point(atoms_1)
    centre_point_2 = cell_centre_point(atoms_2)

    atoms_3 = deepcopy(atoms_2)
    new_positions = mat(get_positions(atoms_3))
    new_positions[1,:] += centre_point_1[1]-centre_point_2[1]
    new_positions[2,:] += centre_point_1[2]-centre_point_2[2]
    set_positions!(atoms_3, new_positions)
    set_cell!(atoms_3, get_cell(atoms_1))

    return atoms_3
end

"""
Generates crack system from a bulk using matscipy isotropic_modeI_crack_tip_displacement_field
"""
function setup_crack(atoms_bulk, tip, k1, k1g, C44, nu)

    @pyimport matscipy.fracture_mechanics.crack as crack

    atoms_crack = deepcopy(atoms_bulk)
    r, t = calc_radial_positions_xy(atoms_crack, tip)
    u, v = crack.isotropic_modeI_crack_tip_displacement_field(k1*k1g, C44, nu, r, t)
    positions_temp = (mat(get_positions(atoms_crack)))
    positions_temp[1,:] = positions_temp[1,:] + u
    positions_temp[2,:] = positions_temp[2,:] + v
    set_positions!(atoms_crack, positions_temp)

    return atoms_crack
end

function surface_energy(atoms, calculator)

    set_calculator!(atoms, calculator)
    atoms_surface = deepcopy(atoms)

    vacuum = 20
    shift = vacuum/2

    pos = mat(get_positions(atoms_surface))
    pos[2,:] += shift
    set_positions!(atoms_surface, pos)

    cell = get_cell(atoms_surface)
    cell[5] += vacuum
    set_cell!(atoms_surface, cell)

    e0 = potential_energy(atoms)
    e1 = potential_energy(atoms_surface)

    gamma_se = (e1 - e0)/(get_cell(atoms)[1])
    surface_energy = gamma_se * 10 #  in GPa*A = 0.1 J/m^2

    return surface_energy
end



# ----- plot atoms -----

"""
    plot_atoms(atoms, ; colour="b", indices=nothing, scale=.1, cell=false)

Plot xy positions of atoms

#### Arguments
- indices : default will plot all, provide subset, ie [153, 154] to plot subset only
- cell : draw a box in xy plane repesenting the cell
"""
function plot_atoms(atoms, ; colour="b", indices=nothing, scale=.1, cell=false)

    if indices == nothing
        indices = linearindices(atoms)
    end

    p = mat(get_positions(atoms))
    scatter(p[1,indices], p[2,indices], c=colour, s=scale)

    axis(:equal)

    if cell == true
      cell_a = get_cell(atoms)
      vlines(0, 0, cell_a[5], color="black", alpha=0.2)
      vlines(cell_a[1], 0, cell_a[5], color="black", alpha=0.2)
      hlines(0, 0, cell_a[1], color="black", alpha=0.2)
      hlines(cell_a[5], 0, cell_a[1], color="black", alpha=0.2)
    end

end



end
