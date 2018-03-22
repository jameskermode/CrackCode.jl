# ----- plot atoms -----

# basic plotting of atoms

module Plot

using JuLIP: get_positions, mat, get_cell
using PyPlot: plot, scatter, axis, vlines, hlines, legend, title
using ASE

"""
`box_around_point(point, box_width)`

2D (x,y) limits of a box, centred on `point` of width `box_width`

### Usage
`box_around_point(tip, [20,10])`

Generally used in
`axis(box_around_point(tip, [20,10]))`

### Arguements
- `point`: eg [x, y, z]
- `box_width`: [width of x, width of y]

### Returns
`x_min`, `x_max`, `y_min`, `y_max`

"""
# Not much need for this function, can just write the limits
# especially if its only used in axis
function box_around_point(point, box_width)

    x_c = point[1]
    y_c = point[2]

    box_width_x = box_width[1]
    box_width_y = box_width[2]

    x_min = x_c - box_width_x
    x_max = x_c + box_width_x

    y_min = y_c - box_width_y
    y_max = y_c + box_width_y

    return x_min, x_max, y_min, y_max
end

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

        # check if the BoundaryConditions module is imported
        if isdefined(:BoundaryConditions) == true
            if indices != linearindices(atoms)
                b_i = []
                for a_i in indices
                    bonds_list_a_i, mB_a_i, list_a_i = BoundaryConditions.get_bonds(atoms, a_i, bonds_list = bonds_list)
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


# plot for the function
# find_next_bond_along(atoms, bonds_list, a0, tip, tip_new; plot=false)

"""
`plot_next_bond_along(atoms, a0, tip, tip_next, across_crack, across_crack_next)`

2D (x,y) plot for the function `find_next_bond_along(atoms, bonds_list, a0, tip, tip_new)`
a visual check for the above function to make sure its selected the correct bond

### Arguments
- `atoms`: Atoms object
- `a0`: crystal lattice constant
- `tip`: current tip vector
- `tip_next`: next tip vector
- `across_crack`: bonds that cross the crack behind `tip`, comes from `BoundaryConditions.filter_crack_bonds()`
- `across_crack_next`: bond that crosses the crack behind `tip_next`, comes from `find_next_bond_along()`
"""
function plot_next_bond_along(atoms, a0, tip, tip_next, across_crack, across_crack_next)

    # poor way of doing it as if BoundaryConditions.find_next_bond_along()
    # changes this section will need to be change too

    # recalculate some values to plot
    # variables: nearby, a1
    # section: from BoundaryConditions.find_next_bond_along()
    pos = get_positions(atoms)
    radial_distances = norm.(pos .- tip)

    # all atoms near the current (given) crack tip
    nearby = find( radial_distances .< a0 )

    # of the nearby list find the atom with the closest distance from the tip
    distances = zeros(length(nearby))
    for i in 1:length(nearby)
        distances[i] = norm(tip[1] - atoms[nearby[i]])
    end
    index = find(distances .== minimum(distances))
    a1 = nearby[index][1]
    # section: end

    scatter(tip[1][1], tip[1][2], color="red", s=8, label="tip")
    scatter(tip_next[1][1], tip_next[1][2], color="purple", s=8, label="tip next")

    plot_atoms(atoms)

    plot_atoms(atoms, indices=nearby, colour="blue", scale=5, label="nearby to tip")
    plot_circle(radius=a0, centre=tip[1], colour="grey", label="nearby radius")

    plot_atoms(atoms, indices=a1, colour="green", scale=10, label="chosen nearby atom")

    plot_bonds(atoms, across_crack, linewidth=0.5, label="across crack")
    plot_bonds(atoms, [across_crack_next], linewidth=2.0, label="next bond")

    axis(box_around_point([tip[1][1], tip[1][2]], [2.5*a0,2.5*a0]))
    title("Visual Check: Next bond along that was chosen")
    legend()

end




end
