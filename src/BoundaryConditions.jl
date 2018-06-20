module BoundaryConditions






### Old code

#using JuLIP
#using PyCall

using JuLIP: AbstractAtoms, mat, vecs, set_pbc!, get_pbc, get_positions, set_positions!, get_cell, set_cell!,
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



# Could even remove the abstraction of concept of crack
# have it as a line, point etc

"""
    crosses_crack(atoms, i, j, tip)

Decide whether a bond crosses the line to where the crack tip is.
Returns boolean, true if it DOES cross the crack line.
"""
function crosses_crack(atoms, i::Int, j::Int, tip)

    pos = get_positions(atoms)
    p1 = pos[i]
    p2 = pos[j]
    tip = tip[1]

    crack_bond = false
    # selects left hand side of tip
    if (p1[1] < tip[1]) | (p2[1] < tip[1])
        # check if atom 1 is above and atom 2 is below
        # or if atom 2 is above and atom 1 is below
        if ((p1[2] > tip[2]) & (p2[2] < tip[2])) | ((p1[2] < tip[2]) & (p2[2] > tip[2]))
            # multiple indices in the list mean, more than one bond
            crack_bond = true
        end
    end
    return crack_bond
end

"""
Compare one atom, `i`, to several atoms `j`
i.e. which atoms connected to `i` crosses the crack
"""
function crosses_crack(atoms, i::Int, j::Array, tip)

    crack_bonds = []
    for _j in j
        push!(crack_bonds, BoundaryConditions.crosses_crack(atoms, i, _j, tip))
    end

    return crack_bonds
end



"""
    filter_crack_bonds(atoms::AbstractAtoms, bonds_list, crack_tip)

Returns new `bonds_list` and new `mB` with the bonds removed that crossed the crack.
Also returns `across_crack` list of bonds pairs that were removed.

"""
# the filter part was a separate function 'remove_bonds'
# generic and easy to remember but now one line of code, bit pointless
function filter_crack_bonds(atoms::AbstractAtoms, bonds_list, crack_tip)

    across_crack = Array{Tuple{Int, Int}}(0)
    for b in bonds_list
        i = b[1]
        j = b[2]

        if (crosses_crack(atoms, i, j, crack_tip) == true)
            push!(across_crack, (i,j))
        end
    end

    bonds_list = filter(array -> array ∉ across_crack, bonds_list)

    P = get_positions(atoms)
    mB = [ 0.5*(P[b[1]] + P[b[2]])  for b in bonds_list]

    return bonds_list, mB, across_crack
end

"""
`find_next_bond_along(atoms, bonds_list, a0, tip, tip_new)`

Find the bond that is beyond the crack tip

### Arguments
- `atoms`: Atoms object
- `bonds_list`: list of bond tuples
- `a0`: lattice spacing
- `tip`: current tip position
- `tip_new`: next tip position, advanced by one bond
- `plot=false`: plot to visually show if it worked

### Notes
- not always guarantee to be one bond, fix

"""
function find_next_bond_along(atoms, bonds_list, a0, tip, tip_new)

    pos = get_positions(atoms)
    radial_distances = norm.(pos .- tip)

    # all atoms near the current (given) crack tip
    nearby = find( radial_distances .< a0 )

    # of the nearby list find the atom with the closest distance from the tip
    distances = zeros(length(nearby))
    pos = get_positions(atoms)
    for i in 1:length(nearby)
        distances[i] = norm(tip[1] - pos[nearby[i]])
    end
    index = find(distances .== minimum(distances))
    a1 = nearby[index][1]

    # list of atoms bonded to a1
    bonds_list_a1, mB_a1, list_a1 = BoundaryConditions.get_bonds(atoms, a1, bonds_list = bonds_list)


    # Note:
    #   - technically don't need above section, if you provided a bonds_list with crack bonds already removed
    #   - next two lines should just find the next bond along
    #   - top section only limits the selection of bonds to the region near the crack tip
    #       - needed if bonds_list is complete
    #   - crack in single line, x, this might be fine, might need something like this for crack that moves in plane
    #   - currently only works in x, would need to change filter_crack_bonds as that only works in x too

    # likely to not generalise to other systems, might get more than one that crosses

    # should sperate this part out?
    # as filter_crack_bonds is already a function then the find_next_bond_along()
    # algorithim is by it self, maybe need to change the name

    # next bond (hopfully just one bond) along the crack tip
    _, _, across_crack = BoundaryConditions.filter_crack_bonds(atoms, bonds_list, tip_new)



    bond = across_crack

    return bond

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
    u_cle_old(atoms, tip; nu = 0.25)

Displacement field, Continuum Linear Elastic solution.
Returns displacements in cartesian coordinates.

Isotropic crack solution:

``u_x = C √r [ (2κ - 1)cos(θ/2) - cos(3θ/2) ]``

``u_y = C √r [ (2κ + 1)sin(θ/2) - sin(3θ/2) ]``

``C = (K_I √(2π)) / (8μπ)``

``κ = 3 - 4nu``

where the formula for ``κ `` is the one for plane strain
(plane stress is different) and ``nu`` is Poisson ratio, i.e., ``nu = 1/4``
and hence ``κ = 2``
"""
function u_cle_old(atoms, tip; nu = 0.25)

    pos = mat(get_positions(atoms) .- tip)

    x = pos[1,:]
    y = pos[2,:]

    κ = 3 - 4 * nu
    r = sqrt.(x.^2 + y.^2)
    θ = angle.(x + im * y)
    ux = sqrt.(r) .* ((2*κ-1) * cos.(θ/2) - cos.(3*θ/2))
    uy = sqrt.(r) .* ((2*κ+1) * sin.(θ/2) - sin.(3*θ/2))
    uz = zeros(length(ux))
    return vecs([ux'; uy'; uz'])
end


"""
    u_cle(atoms, tip, K, E, nu)

Displacement field, Continuum Linear Elastic solution.
Returns displacements in cartesian coordinates.

Isotropic crack solution:

``u_x = C √r [ (2κ - 1)cos(θ/2) - cos(3θ/2) ]``

``u_y = C √r [ (2κ + 1)sin(θ/2) - sin(3θ/2) ]``

``C = K / (2√(2pi)E))*(1+nu)``

``κ = 3 - 4nu``

where the formula for ``κ `` is the one for plane strain
(plane stress is different), ``nu`` is Poisson ratio and
E is the Youngs Modulus
"""
function u_cle(atoms, tip, K, E, nu)

    pos = mat(get_positions(atoms) .- tip)

    x = pos[1,:]
    y = pos[2,:]

    κ = 3 - 4 * nu
    r = sqrt.(x.^2 + y.^2)
    θ = angle.(x + im * y)
    C = (K / (2*√(2*pi)*E))*(1+nu)

    ux = C*sqrt.(r) .* ((2*κ-1) * cos.(θ/2) - cos.(3*θ/2))
    uy = C*sqrt.(r) .* ((2*κ+1) * sin.(θ/2) - sin.(3*θ/2))
    uz = zeros(length(ux))

    return vecs([ux'; uy'; uz'])
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
`hessian_correction(u, H, f, Idof)`

Calculate the corrected displacements, from solving H \ f.
# Returns
- `u_c`: corrected displacements
# Arguements:
- `u`: initial displacements
- `H`: hessian
- `f`: forces
- `Idof`: degrees of freedom
"""
function hessian_correction(u, H, f, Idof)

    u_c = mat(copy(u))[:]
    f_c = mat(copy(f))[:]

    u_c[Idof] -= H[Idof, Idof] \ -f_c[Idof]

    u_c = vecs(u_c)

    return u_c
end

"""
`hessian_correction(atoms::AbstractAtoms, u, Idof; H = nothing, steps = 1)`

Calculate the corrected displacements on an atoms objects, from solving H \ f.
Able to perform multiple hessian correction steps

# Returns
- `u_step`: corrected displacements
# Arguements:
- `atoms`: atoms object
- `u`: initial displacements (Cartesian)
- `Idof`: degrees of freedom #Fix this: not ideal to have cartensian and dof vector
## Optional Arguements:
- `H = nothing`: provide hessian, defaults to calculating hessian of atoms object
- `steps = 1`: number of hessian correction steps to compute
"""
function hessian_correction(atoms::AbstractAtoms, u, Idof; H = nothing, steps = 1)

    u_c = []
    u_step = u # for step == 1

    for step in 1:steps

        pos_original = get_positions(atoms)
        set_positions!(atoms, pos_original + u_step)

        if H == nothing
            H = hessian(atoms)
        end
        f = forces(atoms)

        u_step = hessian_correction(u_step, H, f, Idof)

        set_positions!(atoms, pos_original)

        u_c = u_step
    end

    return u_c
end

"""
`u_relaxed(atoms, u, constraints)`

Minimise atoms and return relaxed displacements, u_r.
# Returns
- `u_r`: relaxed displacements
# Arguements
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
# Arguements
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
