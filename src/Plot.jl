# ----- plot atoms -----

# basic plotting of atoms

module plot

using JuLIP


"""
    plot_bonds(atoms, bonds_list; indices=nothing, colour="grey", alpha=0.5, linewidth=0.5)

Plot bonds of given pairs

#### Arguments
- atoms: ASEAtoms
- bonds_list: draw bonds of given indices pairs
- indices : default will plot all, provide subset, ie [153, 154] to plot subset only
- colour
- alpha: transprancy channel
- linewidth
"""
function plot_bonds(atoms, bonds_list; indices=nothing, colour="grey", alpha=0.5, linewidth=0.5)

    # semi-redundent because can just filter bonds_list pretty easily in input
    # consistent with atoms plotting though, can pass the same indices

    if indices != nothing
        bonds_list = bonds_list[indices]
    end

    pos = get_positions(atoms)
    for b in bonds_list
        plot([pos[b[1]][1], pos[b[2]][1]], [pos[b[1]][2], [pos[b[2]][2]]], color=colour, alpha=0.5, linewidth=linewidth, linestyle="--")
    end

end

"""
    plot_atoms(atoms; indices=nothing, bonds_list = nothing, cell=false,
                    colour="b", scale=.1, )

Plot xy positions of atoms

#### Arguments
- atoms: ASEAtoms
- indices : default will plot all, provide subset, ie [153, 154] to plot subset only
- bonds_list: draw bonds of given indices pairs
- cell : draw a box in xy plane repesenting the cell
- colour
- scale: size of atoms
"""
function plot_atoms(atoms; indices=nothing, bonds_list = nothing, cell=false, colour="b", scale=.1, )

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

    if bonds_list != nothing
        if indices == linearindices(atoms)
            plot_bonds(atoms, bonds_list)
        end

        # check if the boundary_conditions module is imported
        if isdefined(:boundary_conditions) == true
            if indices != linearindices(atoms)
                b_i = []
                for a_i in indices
                    bonds_list_a_i, mB_a_i, list_a_i = boundary_conditions.get_bonds(atoms, a_i, bonds_list = bonds_list)
                    append!(b_i, list_a_i)
                end
                plot_bonds(atoms, bonds_list; indices=b_i)
            end
        end
    end

end

end
