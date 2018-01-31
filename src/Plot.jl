# ----- plot atoms -----

# basic plotting of atoms

module Plot

using JuLIP: get_positions, mat
using PyPlot: plot, scatter, axis, vlines, hlines

"""
`plot_bonds(atoms, bonds_list; indices=nothing, colour="grey", alpha=0.5,
                                                    linewidth=0.5, label="")`

Plot bonds of given pairs

### Arguments
- `atoms`: ASEAtoms
- `bonds_list`: draw bonds of given indices pairs
- `indices`: plot subset, default will plot all
- `colour`
- `alpha`: transprancy channel
- `linewidth`
- `label`
"""
function plot_bonds(atoms, bonds_list; indices=nothing, colour="grey", alpha=0.5, linewidth=0.5, label="")

    # semi-redundent because can just filter bonds_list pretty easily in input
    # consistent with atoms plotting though, can pass the same indices

    if indices != nothing
        bonds_list = bonds_list[indices]
    end

    pos = get_positions(atoms)
    for b in bonds_list
        plot([pos[b[1]][1], pos[b[2]][1]], [pos[b[1]][2], [pos[b[2]][2]]],
                color=colour, alpha=0.5, linewidth=linewidth, linestyle="--")
    end

    # plot last one again with a label
    # TODO: find better way to do this
    b = bonds_list[length(bonds_list)]
    plot([pos[b[1]][1], pos[b[2]][1]], [pos[b[1]][2], [pos[b[2]][2]]],
            color=colour, alpha=0.5, linewidth=linewidth, linestyle="--", label=label)


end

"""
`plot_atoms(atoms; indices=nothing, bonds_list = nothing, cell=false,
                                                colour="b", scale=.1, label="")`

Plot xy positions of atoms

### Arguments
- `atoms`: ASEAtoms
- `indices` : plot subset, default will plot all
- `bonds_list`: draw bonds of given indices pairs
- `cell`: draw a box in xy plane repesenting the cell
- `colour`
- `scale`: size of atoms
- `label`
"""
function plot_atoms(atoms; indices=nothing, bonds_list = nothing, cell=false, colour="b", scale=.1, label="")

    if indices == nothing
        indices = linearindices(atoms)
    end

    p = mat(get_positions(atoms))
    scatter(p[1,indices], p[2,indices], c=colour, s=scale, label=label)

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
            plot_bonds(atoms, bonds_list, label=string(label, " - atoms_bonds"))
        end

        # check if the boundary_conditions module is imported
        if isdefined(:boundary_conditions) == true
            if indices != linearindices(atoms)
                b_i = []
                for a_i in indices
                    bonds_list_a_i, mB_a_i, list_a_i = boundary_conditions.get_bonds(atoms, a_i, bonds_list = bonds_list)
                    append!(b_i, list_a_i)
                end
                plot_bonds(atoms, bonds_list; indices=b_i, label=string(label, " - atoms_bonds"))
            end
        end
    end

end

"""
`plot_circle(;radius=1.0, centre=[0.0,0.0,0.0], colour="blue", linewidth=1.0,
                                                    linestyle="--", label="")`

Plot circle of given radius around a given point

### Arguments
- `radius = 1.0`: radius of circle
- `centre = [0.0,0.0]`: x y positions at which to centre the circle
- `colour`
- `linewidth`
- `linestyle`
- `label`
"""
function plot_circle(;radius=1.0, centre=[0.0,0.0,0.0], colour="blue",
                                                linewidth=1.0, linestyle="--", label="")

    interval_width = pi/20.0

    x = radius.*cos.(-pi:interval_width:pi) .+ centre[1]
    y = radius.*sin.(-pi:interval_width:pi) .+ centre[2]

    plot(x , y, color=colour, linestyle=linestyle, linewidth=linewidth, label=label)
end

end
