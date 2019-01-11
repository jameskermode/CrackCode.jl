# Basic plotting of atoms objects
# and other atoms object properties

module Plot

    using JuLIP: JVecF, Atoms, get_positions, mat, get_cell
    using PyPlot: plot, subplots, scatter, axis, vlines, hlines, legend, title

    export plot_next_bond_along, plot_atoms, plot_bonds, box_around_point, plot_mask

    """
    `plot_next_bond_along(atoms::Atoms, tip::JVecF, across_crack_behind::Array{Tuple{Int, Int}}, 
    across_crack_ahead::Array{Tuple{Int, Int}}, next_bonds::Array{Tuple{Int, Int}})`

    2D (x,y) plot for a visual check of picking the next bond.
    Need the arrays from sub module BoundaryConditions eg;
    - `bonds_list, mB, across_crack = filter_crack_bonds(atoms, bonds_list, normal_crack_plane, tip, normal_crack_front, tip)
    - `next_bonds, across_crack_ahead = find_next_bond_along2(atoms, bonds_list, normal_crack_plane, tip, normal_crack_front, tip, verbose = 1)`
    then 
    - plot_next_bond_along(atoms, tip, across_crack, across_crack_ahead, next_bonds)
    """
    function plot_next_bonds_along(atoms::Atoms, tip::JVecF, across_crack_behind::Array{Tuple{Int, Int}},
                                                                            across_crack_ahead::Array{Tuple{Int, Int}}, next_bonds::Array{Tuple{Int, Int}})
        plot_atoms(atoms, scale=2, colour="grey")
        scatter(tip[1], tip[2], color="b", s=8, label="tip")
        plot_bonds(atoms, across_crack_behind, colour = "grey", label="pairs: across crack and behind and on tip")
        plot_bonds(atoms, across_crack_ahead, colour = "r", label="pairs: across crack and ahead tip")
        plot_bonds(atoms, next_bonds, colour = "g", linewidth=1, label="pair(s): next_bond(s)")

        dis = norm(atoms[across_crack_ahead[1][2]] - atoms[across_crack_ahead[1][1]]) # get some distance to only zoom in
        axis(box_around_point([tip[1], tip[2]], [2.5*dis,2.5*dis]))
        legend()
    end

    """
    `plot_mask(atoms::Atoms, mask::Array{Bool}; dim::Symbol=:x)`

    ### Arguments
    - `atoms` : Atoms object
    - `mask` : boolean matrix 3xN

    ### Optional Arguments
    - `dim=:x` : dimension of mask to plot, either :x, :y or :z
    """
    function plot_mask(atoms::Atoms, mask::Array{Bool}; dim::Symbol=:x)

        dim_labels = [:x, :y, :z]
        i = find(dim .== dim_labels)[1]

        fig, ax = subplots(1,1)
        plot_atoms(atoms, indices=find(mask[i,:] .== true), colour="green", label="free")
        plot_atoms(atoms, indices=find(mask[i,:] .== false), colour="red", label="clamped")
        ax[:legend]()
        ax[:set_title]("mask axis: $(dim)")

        return fig, ax
    end


# Old code

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
- `alpah`: transparency
- `label`
"""
function plot_atoms(atoms; indices=nothing, bonds_list = nothing, cell=false, colour="b", scale=.1, alpha=1, label="")

    if indices == nothing
        indices = linearindices(atoms)
    end

    p = mat(get_positions(atoms))
    scatter(p[1,indices], p[2,indices], c=colour, s=scale, alpha=alpha, label=label)

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



end
