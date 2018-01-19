
module correction

using JuLIP
using PyPlot

using JuLIP.ASE.MatSciPy.NeighbourList

function calc_mapping_list(atoms_1, atoms_2)

    # assumes atoms are aligned how you want them
    # in this case atoms 1 is the larger system
    atoms_1 = deepcopy(atoms_1)
    atoms_2 = deepcopy(atoms_2)

    length_atoms_1 = length(atoms_1)

    map_symbols = fill("C", length(atoms_2))
    atoms_2.po["set_chemical_symbols"](map_symbols)
    atoms_2.po["get_chemical_symbols"]();


    atoms_combined = atoms_1.po["extend"](atoms_2.po)
    atoms_combined = JuLIP.ASE.ASEAtoms(atoms_combined)

    nb_list = NeighbourList(atoms_combined, 0.001)

    indices_in_ref = nb_list.i[nb_list.i .< length_atoms_1];

    return indices_in_ref

end


function calc_correction(atoms)

    h = hessian(atoms)
    correction_vector = h \ -gradient(atoms)

    correction = zeros(3, length(atoms))
    correction[constraint(atoms).ifree] = correction_vector

    return correction
end


function reduced_xyz_array(array, indices)

    array_reduced = zeros(3, length(indices))

    for i in 1:length(indices)
        array_reduced[:,i] = array[:,indices[i]]

    end

    return array_reduced
end


function plot_correction(atoms, correction)

  pos =  mat(get_positions(atoms))

  quiver(pos[1,:], pos[2,:], correction[1,:], correction[2,:])
  axis(:equal)

end


end
