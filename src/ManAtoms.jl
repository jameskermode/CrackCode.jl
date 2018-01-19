# Manlipulation of atoms object
# or other atoms object properties

module manatoms

using JuLIP
using PyCall
using PyPlot

using JuLIP.ASE.MatSciPy.NeighbourList

#export centre_point

include("generic.jl")

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
    dimer(element ; seperation=1.0, cell_size=30.0)

Build a dimer object with specifc seperation and cell size
#### Arguments
- element : chemical element
- (optional keyword) seperation : distance between atom along x axis
- (optional keyword) cell_size : float, same in all directions
"""

function dimer(element="Si", ; seperation=1.0, cell_size=30.0)

    a_string = join([element, "2"])
    atoms = ASEAtoms(a_string)
    set_cell!(atoms, cell_size*eye(3))
    set_pbc!(atoms, true)
    positions = mat(get_positions(atoms))

    positions[1,1] += -seperation/2
    positions[1,2] += +seperation/2

    set_positions!(atoms, positions)

    return atoms
end

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
      positions = mat(get_positions(atoms))
      # mid_point in height between pair is 0.5 ( a0 * cos(pi/6) )
      positions[2,:] += a*sqrt(3)/4
      set_positions!(atoms, positions)
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

    x = mat(positions(atoms))[1, :]
    y = mat(positions(atoms))[2, :]
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

    positions = mat(get_positions(atoms))
    pos_i1 = positions[:,index_1]
    pos_i2 = positions[:,index_2]
    seperation = sqrt((pos_i1[1]-pos_i2[1])^2 + (pos_i1[2]-pos_i2[2])^2 + (pos_i1[3]-pos_i2[3])^2)

    return seperation
end

function get_size(atoms)

    positions = mat(get_positions(atoms))

    x_length = maximum(positions[1,:]) - minimum(positions[1,:])
    y_length = maximum(positions[2,:]) - minimum(positions[2,:])
    z_length = maximum(positions[3,:]) - minimum(positions[3,:])

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
    positions_temp = (mat(positions(atoms_crack)))
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

    p = mat(positions(atoms))
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
